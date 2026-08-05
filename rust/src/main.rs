// A genuinely tiny language model — word-level this time.
//
// v1 of this file predicted the next CHARACTER from the current one: an
// order-1 Markov chain wearing a neural net costume. No amount of training
// could make that produce real words, because it had no concept of "word"
// at all — every character decision was independent of everything more
// than one character back.
//
// This version fixes that at the architecture level: tokens are WORDS (and
// punctuation), and each prediction is conditioned on a window of the last
// N tokens, not just one. Concretely, this is a small instance of the
// classic 2003 Bengio neural language model:
//
//   [tok(t-N) ... tok(t-1)] --embed--> concat --W1,tanh--> hidden
//                                                --W2,softmax--> P(tok t)
//
// Same real backprop as before, just through two weight matrices and a
// context window instead of one. Trained on real text, it should produce
// actual dictionary words in locally-plausible order — readable, often
// nonsensical, but real words, which the character model could never do
// no matter how long you trained it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A genuinely tiny word-level language model, trained via real backprop.
#[derive(Parser, Debug)]
#[command(name = "tiny_llm", about, long_about = None)]
struct Cli {
    /// Training text file to learn from (default: training-data.txt next to this crate)
    #[arg(short = 'd', long = "data-set", value_name = "PATH")]
    data_set: Option<PathBuf>,

    /// Training epochs
    #[arg(short = 'e', long, default_value_t = 10)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.1)]
    lr: f32,

    /// Tokens to generate per sample
    #[arg(short = 'l', long, default_value_t = 80)]
    length: usize,

    /// Truncate the data set to the first N characters (handy for large files)
    #[arg(long, value_name = "N")]
    max_chars: Option<usize>,

    /// Vocabulary size cap: keeps the N most frequent tokens, everything
    /// else collapses to <unk>. Bounds the (expensive) output layer.
    #[arg(long, default_value_t = 3000)]
    vocab_size: usize,

    /// Context window: how many previous tokens condition each prediction.
    /// This is what lets the model see more than one token of history —
    /// the whole reason it can learn words/phrases instead of just letters.
    #[arg(long, default_value_t = 4)]
    context: usize,

    /// Embedding dimension (size of each token's learned vector)
    #[arg(long, default_value_t = 16)]
    embed_dim: usize,

    /// Hidden layer size (the MLP between context and output layer)
    #[arg(long, default_value_t = 48)]
    hidden_dim: usize,

    /// Pairs per mini-batch. Each batch's gradients are computed in parallel
    /// across cores, then applied as one averaged update. Bigger = faster
    /// (better parallelism) but noisier/less faithful to per-example SGD.
    #[arg(short = 'b', long, default_value_t = 8192)]
    batch_size: usize,

    /// Load a previously trained model checkpoint instead of starting from
    /// random weights. Its vocab/context/embed_dim/hidden_dim win over the
    /// CLI flags above (they're baked into the checkpoint).
    #[arg(long, value_name = "PATH")]
    load_model: Option<PathBuf>,

    /// Save the trained model to this path after training (or immediately,
    /// if --load-model was given and --epochs is 0).
    #[arg(long, value_name = "PATH")]
    save_model: Option<PathBuf>,
}

// ---------- Tokenizer ----------
// Splits into: runs of letters/apostrophes as one word token ("thou'lt"),
// each other punctuation character as its own token, and '\n' as its own
// token (so the model can learn *where lines end* — the closest thing to
// verse structure it can pick up). Plain whitespace is just a separator.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphabetic() || c == '\'' {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            tokens.push(std::mem::take(&mut word));
        }
        if c == '\n' {
            tokens.push("\n".to_string());
        } else if !c.is_whitespace() {
            tokens.push(c.to_string());
        }
    }
    if !word.is_empty() {
        tokens.push(word);
    }
    tokens
}

// Punctuation that hugs the token before it, with no space in between.
fn attaches_to_previous(tok: &str) -> bool {
    matches!(tok, "," | "." | "!" | "?" | ";" | ":" | "'" | ")" | "]" | "}" | "-")
}

// Reconstruct readable text from a token stream. Heuristic, not a real
// detokenizer, but good enough to make output look like actual prose/verse
// instead of a token-per-line dump.
fn detokenize(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut prev: Option<&str> = None;
    for tok in tokens {
        if tok == "\n" {
            out.push('\n');
        } else {
            let need_space = match prev {
                None => false,
                Some("\n") => false,
                Some(p) if matches!(p, "(" | "[" | "{" | "`") => false,
                _ if attaches_to_previous(tok) => false,
                _ => true,
            };
            if need_space {
                out.push(' ');
            }
            out.push_str(tok);
        }
        prev = Some(tok.as_str());
    }
    out
}

// Keep the `max_vocab - 1` most frequent tokens (reserving slot 0 for
// <unk>), then sort alphabetically for a readable listing. Everything not
// in this vocab maps to <unk> at lookup time — standard for bounding an
// output layer's cost against a Zipfian real-text vocabulary.
fn build_vocab(tokens: &[String], max_vocab: usize) -> (Vec<String>, HashMap<String, usize>) {
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        *freq.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut counted: Vec<(&str, usize)> = freq.into_iter().collect();
    counted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let keep = max_vocab.saturating_sub(1);
    let mut top: Vec<String> = counted.into_iter().take(keep).map(|(s, _)| s.to_string()).collect();
    top.sort();

    let mut vocab = Vec::with_capacity(top.len() + 1);
    vocab.push("<unk>".to_string());
    vocab.append(&mut top);

    let token_to_id = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    (vocab, token_to_id)
}

// ---------- Model parameters ----------
// [tok(t-N)..tok(t-1)] --embed+concat--> W1,tanh --> hidden --> W2,softmax --> P(next tok)
struct Model {
    vocab_size: usize,
    embed_dim: usize,
    context: usize,
    hidden_dim: usize,

    embedding: Vec<f32>, // [token * embed_dim + k]
    w1: Vec<f32>,        // [(context*embed_dim) * hidden_dim], row-major over input index
    b1: Vec<f32>,        // [hidden_dim]
    w2: Vec<f32>,        // [hidden_dim * vocab_size], row-major over hidden index
    b2: Vec<f32>,        // [vocab_size]
}

// Gradients accumulated over a batch (or a chunk of one), mirroring the
// parameter shapes above, plus the summed loss for reporting.
struct Grad {
    d_embedding: Vec<f32>,
    d_w1: Vec<f32>,
    d_b1: Vec<f32>,
    d_w2: Vec<f32>,
    d_b2: Vec<f32>,
    loss: f32,
}

impl Grad {
    fn zeros(vocab_size: usize, embed_dim: usize, context: usize, hidden_dim: usize) -> Self {
        Grad {
            d_embedding: vec![0.0; vocab_size * embed_dim],
            d_w1: vec![0.0; context * embed_dim * hidden_dim],
            d_b1: vec![0.0; hidden_dim],
            d_w2: vec![0.0; hidden_dim * vocab_size],
            d_b2: vec![0.0; vocab_size],
            loss: 0.0,
        }
    }

    fn add_assign(&mut self, other: &Grad) {
        for (a, b) in self.d_embedding.iter_mut().zip(&other.d_embedding) {
            *a += b;
        }
        for (a, b) in self.d_w1.iter_mut().zip(&other.d_w1) {
            *a += b;
        }
        for (a, b) in self.d_b1.iter_mut().zip(&other.d_b1) {
            *a += b;
        }
        for (a, b) in self.d_w2.iter_mut().zip(&other.d_w2) {
            *a += b;
        }
        for (a, b) in self.d_b2.iter_mut().zip(&other.d_b2) {
            *a += b;
        }
        self.loss += other.loss;
    }
}

// A trained model plus everything needed to make its token ids meaningful
// again later — this whole struct is exactly what "saving the model" means.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    vocab: Vec<String>,
    context: usize,
    embed_dim: usize,
    hidden_dim: usize,
    embedding: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl Model {
    fn new(vocab_size: usize, embed_dim: usize, context: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();
        // Roughly Xavier-ish init: scale by 1/sqrt(fan_in) per layer so the
        // tanh hidden layer starts in its responsive range instead of
        // saturated or vanishing.
        let mut rand_vec = |len: usize, fan_in: usize| -> Vec<f32> {
            let scale = 1.0 / (fan_in as f32).sqrt();
            (0..len).map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale).collect()
        };
        Model {
            vocab_size,
            embed_dim,
            context,
            hidden_dim,
            embedding: rand_vec(vocab_size * embed_dim, embed_dim),
            w1: rand_vec(context * embed_dim * hidden_dim, context * embed_dim),
            b1: vec![0.0; hidden_dim],
            w2: rand_vec(hidden_dim * vocab_size, hidden_dim),
            b2: vec![0.0; vocab_size],
        }
    }

    fn from_checkpoint(ckpt: Checkpoint) -> (Self, Vec<String>) {
        let vocab_size = ckpt.vocab.len();
        let model = Model {
            vocab_size,
            embed_dim: ckpt.embed_dim,
            context: ckpt.context,
            hidden_dim: ckpt.hidden_dim,
            embedding: ckpt.embedding,
            w1: ckpt.w1,
            b1: ckpt.b1,
            w2: ckpt.w2,
            b2: ckpt.b2,
        };
        (model, ckpt.vocab)
    }

    fn to_checkpoint(&self, vocab: &[String]) -> Checkpoint {
        Checkpoint {
            vocab: vocab.to_vec(),
            context: self.context,
            embed_dim: self.embed_dim,
            hidden_dim: self.hidden_dim,
            embedding: self.embedding.clone(),
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        }
    }

    // ---------- Forward pass ----------
    // Loop order throughout is "outer = row index, inner = contiguous
    // slice" to match each matrix's row-major layout — keeps every
    // inner-loop access contiguous, which is what lets the compiler
    // auto-vectorize instead of chasing scattered cache lines.
    fn forward(&self, ctx: &[usize], concat: &mut [f32], hidden: &mut [f32], logits: &mut [f32], probs: &mut [f32]) {
        let embed_dim = self.embed_dim;
        let hidden_dim = self.hidden_dim;
        let vocab_size = self.vocab_size;

        for (p, &tok) in ctx.iter().enumerate() {
            let src = &self.embedding[tok * embed_dim..tok * embed_dim + embed_dim];
            concat[p * embed_dim..(p + 1) * embed_dim].copy_from_slice(src);
        }

        hidden.copy_from_slice(&self.b1);
        for (in_idx, &x) in concat.iter().enumerate() {
            let w_row = &self.w1[in_idx * hidden_dim..(in_idx + 1) * hidden_dim];
            for h in 0..hidden_dim {
                hidden[h] += x * w_row[h];
            }
        }
        for h in hidden.iter_mut() {
            *h = h.tanh();
        }

        logits.copy_from_slice(&self.b2);
        for (h_idx, &hv) in hidden.iter().enumerate() {
            let w_row = &self.w2[h_idx * vocab_size..(h_idx + 1) * vocab_size];
            for j in 0..vocab_size {
                logits[j] += hv * w_row[j];
            }
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exps = 0.0f32;
        for j in 0..vocab_size {
            let exp = (logits[j] - max_logit).exp();
            probs[j] = exp;
            sum_exps += exp;
        }
        for p in probs.iter_mut() {
            *p /= sum_exps;
        }
    }

    // ---------- Loss + backward pass (real gradients, not approximated) ----------
    // Standard 2-layer MLP backprop: cross-entropy/softmax gradient at the
    // output, through W2 into the tanh hidden layer, through W1 into the
    // concatenated context embeddings, then scattered back into whichever
    // vocabulary rows those context positions actually were.
    //
    // Purely read-only w.r.t. `self` — safe to call from many threads at
    // once, each accumulating into its own `Grad`. That split (read-only
    // parallel gradient computation, then one serial update) is what lets
    // mini-batches parallelize without any unsafe code or locking.
    fn accumulate_gradients(&self, indices: &[usize], example_ctx: &[usize], example_target: &[usize]) -> Grad {
        let embed_dim = self.embed_dim;
        let hidden_dim = self.hidden_dim;
        let vocab_size = self.vocab_size;
        let context = self.context;
        let mut grad = Grad::zeros(vocab_size, embed_dim, context, hidden_dim);

        let mut concat = vec![0.0f32; context * embed_dim];
        let mut hidden = vec![0.0f32; hidden_dim];
        let mut logits = vec![0.0f32; vocab_size];
        let mut probs = vec![0.0f32; vocab_size];
        let mut d_logits = vec![0.0f32; vocab_size];
        let mut d_hidden_pre = vec![0.0f32; hidden_dim];
        let mut d_concat = vec![0.0f32; context * embed_dim];

        for &idx in indices {
            let ctx = &example_ctx[idx * context..(idx + 1) * context];
            let target = example_target[idx];

            self.forward(ctx, &mut concat, &mut hidden, &mut logits, &mut probs);

            d_logits.copy_from_slice(&probs);
            d_logits[target] -= 1.0; // the clean softmax+cross-entropy gradient
            grad.loss += -(probs[target] + 1e-9).ln();

            // Output layer: d_b2, d_W2 (outer product of hidden and d_logits).
            for j in 0..vocab_size {
                grad.d_b2[j] += d_logits[j];
            }
            for h in 0..hidden_dim {
                let hv = hidden[h];
                let dw2_row = &mut grad.d_w2[h * vocab_size..(h + 1) * vocab_size];
                for j in 0..vocab_size {
                    dw2_row[j] += hv * d_logits[j];
                }
            }

            // Backprop through W2 (using its current, pre-update values)
            // into the hidden layer, then through tanh's derivative.
            for h in 0..hidden_dim {
                let w2_row = &self.w2[h * vocab_size..(h + 1) * vocab_size];
                let mut sum = 0.0;
                for j in 0..vocab_size {
                    sum += w2_row[j] * d_logits[j];
                }
                d_hidden_pre[h] = sum * (1.0 - hidden[h] * hidden[h]); // tanh'(x) = 1 - tanh(x)^2
            }

            // Hidden layer: d_b1, d_W1, then backprop into the concatenated
            // context embeddings.
            for h in 0..hidden_dim {
                grad.d_b1[h] += d_hidden_pre[h];
            }
            for in_idx in 0..context * embed_dim {
                let x = concat[in_idx];
                let dw1_row = &mut grad.d_w1[in_idx * hidden_dim..(in_idx + 1) * hidden_dim];
                for h in 0..hidden_dim {
                    dw1_row[h] += x * d_hidden_pre[h];
                }
            }
            for in_idx in 0..context * embed_dim {
                let w1_row = &self.w1[in_idx * hidden_dim..(in_idx + 1) * hidden_dim];
                let mut sum = 0.0;
                for h in 0..hidden_dim {
                    sum += w1_row[h] * d_hidden_pre[h];
                }
                d_concat[in_idx] = sum;
            }

            // Scatter the concat gradient back into the embedding rows for
            // whichever tokens actually filled each context position. Uses
            // +=, so a token repeated across positions/examples correctly
            // accumulates gradient from every occurrence.
            for (p, &tok) in ctx.iter().enumerate() {
                let dst = &mut grad.d_embedding[tok * embed_dim..tok * embed_dim + embed_dim];
                let src = &d_concat[p * embed_dim..(p + 1) * embed_dim];
                for k in 0..embed_dim {
                    dst[k] += src[k];
                }
            }
        }

        grad
    }

    // Apply one averaged update from a batch's accumulated gradients —
    // same "move opposite the gradient" rule as plain SGD, amortized over
    // batch_len examples instead of one.
    fn apply_gradients(&mut self, grad: &Grad, learning_rate: f32, batch_len: usize) {
        let scale = learning_rate / batch_len as f32;
        for (p, g) in self.embedding.iter_mut().zip(&grad.d_embedding) {
            *p -= scale * g;
        }
        for (p, g) in self.w1.iter_mut().zip(&grad.d_w1) {
            *p -= scale * g;
        }
        for (p, g) in self.b1.iter_mut().zip(&grad.d_b1) {
            *p -= scale * g;
        }
        for (p, g) in self.w2.iter_mut().zip(&grad.d_w2) {
            *p -= scale * g;
        }
        for (p, g) in self.b2.iter_mut().zip(&grad.d_b2) {
            *p -= scale * g;
        }
    }
}

// ---------- Generation (autoregressive: feed output back in as next input) ----------
fn generate(model: &Model, vocab: &[String], token_to_id: &HashMap<String, usize>, length: usize) -> String {
    let context = model.context;
    let embed_dim = model.embed_dim;
    let hidden_dim = model.hidden_dim;
    let vocab_size = model.vocab_size;

    // Seed with newline tokens — "start of a line" — so generation begins
    // the way every line in the training data did.
    let newline_id = token_to_id.get("\n").copied().unwrap_or(0);
    let mut ctx: Vec<usize> = vec![newline_id; context];

    let mut rng = rand::rng();
    let mut concat = vec![0.0f32; context * embed_dim];
    let mut hidden = vec![0.0f32; hidden_dim];
    let mut logits = vec![0.0f32; vocab_size];
    let mut probs = vec![0.0f32; vocab_size];
    let mut out_tokens: Vec<String> = Vec::with_capacity(length);

    for _ in 0..length {
        model.forward(&ctx, &mut concat, &mut hidden, &mut logits, &mut probs);

        // Sample from the probability distribution rather than always
        // picking the top choice — this is why LLM output isn't identical
        // every time even for the same prompt.
        let mut r: f32 = rng.random();
        let mut chosen_id = 0;
        for (j, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                chosen_id = j;
                break;
            }
        }

        out_tokens.push(vocab[chosen_id].clone());
        ctx.remove(0);
        ctx.push(chosen_id); // <-- the model's own output becomes its next input
    }

    detokenize(&out_tokens)
}

fn save_checkpoint(path: &PathBuf, model: &Model, vocab: &[String]) -> std::io::Result<()> {
    let ckpt = model.to_checkpoint(vocab);
    let bytes = bincode::serialize(&ckpt).expect("failed to serialize model");
    std::fs::write(path, bytes)
}

fn load_checkpoint(path: &PathBuf) -> Checkpoint {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read model checkpoint {}: {e}", path.display()));
    bincode::deserialize(&bytes).unwrap_or_else(|e| panic!("failed to parse model checkpoint {}: {e}", path.display()))
}

fn main() {
    let cli = Cli::parse();

    // ---------- 1. Load + tokenize ----------
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let default_data_path = PathBuf::from(manifest_dir).join("training-data.txt");
    let data_path = cli.data_set.clone().unwrap_or(default_data_path);

    let fallback_text = "hello world hello there hello friend";
    let mut training_text = match std::fs::read_to_string(&data_path) {
        Ok(text) => {
            println!(
                "Loaded training data from: {} ({} chars)",
                data_path.display(),
                text.chars().count()
            );
            text
        }
        Err(_) => {
            println!(
                "No training file found at {} — using built-in fallback text.",
                data_path.display()
            );
            fallback_text.to_string()
        }
    };

    if let Some(max_chars) = cli.max_chars {
        if training_text.chars().count() > max_chars {
            training_text = training_text.chars().take(max_chars).collect();
            println!("Truncated to --max-chars {} ({} chars)", max_chars, training_text.chars().count());
        }
    }

    let tokens_all = tokenize(&training_text);
    println!("Tokenized into {} tokens", tokens_all.len());

    // ---------- 2. Model: load a checkpoint, or start fresh ----------
    let (vocab, token_to_id, mut model) = if let Some(load_path) = &cli.load_model {
        let ckpt = load_checkpoint(load_path);
        println!(
            "Loaded model from {} (vocab {}, context {}, embed_dim {}, hidden_dim {})",
            load_path.display(),
            ckpt.vocab.len(),
            ckpt.context,
            ckpt.embed_dim,
            ckpt.hidden_dim
        );
        let (model, vocab) = Model::from_checkpoint(ckpt);
        let token_to_id: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        (vocab, token_to_id, model)
    } else {
        let (vocab, token_to_id) = build_vocab(&tokens_all, cli.vocab_size);
        let model = Model::new(vocab.len(), cli.embed_dim, cli.context, cli.hidden_dim);
        (vocab, token_to_id, model)
    };
    println!("Vocab size: {} (includes <unk>)", vocab.len());

    // ---------- 3. Build (context -> next token) examples ----------
    let context = model.context;
    let ids: Vec<usize> = tokens_all.iter().map(|t| token_to_id.get(t.as_str()).copied().unwrap_or(0)).collect();

    let mut example_ctx: Vec<usize> = Vec::new();
    let mut example_target: Vec<usize> = Vec::new();
    if ids.len() > context {
        for i in context..ids.len() {
            example_ctx.extend_from_slice(&ids[i - context..i]);
            example_target.push(ids[i]);
        }
    }
    let num_examples = example_target.len();

    // ---------- 4. Training loop (mini-batch, parallel across cores) ----------
    if cli.epochs > 0 && num_examples > 0 {
        let batch_size = cli.batch_size.max(1);
        let num_threads = rayon::current_num_threads();
        let vocab_size = model.vocab_size;
        let embed_dim = model.embed_dim;

        println!(
            "Training on {} examples (context {}), batch size {}, {} threads available",
            num_examples, context, batch_size, num_threads
        );

        let mut order: Vec<usize> = (0..num_examples).collect();
        let mut rng = rand::rng();

        let start = Instant::now();
        for epoch in 0..cli.epochs {
            order.shuffle(&mut rng); // real training shuffles each epoch; so do we
            let mut total_loss = 0.0f32;

            for batch_start in (0..order.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(order.len());
                let batch_indices = &order[batch_start..batch_end];

                let chunk_len = batch_indices.len().div_ceil(num_threads).max(1);
                let grad = batch_indices
                    .par_chunks(chunk_len)
                    .map(|chunk| model.accumulate_gradients(chunk, &example_ctx, &example_target))
                    .reduce(
                        || Grad::zeros(vocab_size, embed_dim, context, model.hidden_dim),
                        |mut a, b| {
                            a.add_assign(&b);
                            a
                        },
                    );

                total_loss += grad.loss;
                model.apply_gradients(&grad, cli.lr, batch_indices.len());
            }

            if epoch % 1.max(cli.epochs / 20).max(1) == 0 || epoch == cli.epochs - 1 {
                println!("epoch {}: avg loss = {:.4}", epoch, total_loss / num_examples as f32);
            }
        }
        println!("Training took {:.2?}", start.elapsed());
    } else if cli.epochs == 0 {
        println!("--epochs 0: skipping training, using the model as loaded.");
    }

    // ---------- 5. Save, if asked ----------
    if let Some(save_path) = &cli.save_model {
        save_checkpoint(save_path, &model, &vocab).expect("failed to save model checkpoint");
        println!("Saved model to {}", save_path.display());
    }

    // ---------- 6. Generation ----------
    println!("\n--- Generated text ---");
    for _ in 0..3 {
        println!("{}\n---", generate(&model, &vocab, &token_to_id, cli.length));
    }
}
