// A genuinely tiny language model — now with an actual memory.
//
// v1 predicted the next CHARACTER from the current one: an order-1 Markov
// chain. v2 predicted the next WORD from a fixed window of the last 4
// words (a small Bengio-2003-style neural LM) — real words, but no memory
// beyond that fixed window, so no real grammar.
//
// This version replaces the fixed window with a GRU: a hidden state that
// carries information forward across the whole sequence, updated at each
// step by a small gated recurrent cell, trained with real backprop-through-
// time (BPTT). This is the actual architectural lineage that led to modern
// LLMs (RNN -> LSTM/GRU -> Transformer) — the first point in this file's
// history where "how much context can influence a prediction" isn't capped
// by a hardcoded window size.
//
//   h_0 = 0
//   for each token x_t in the sequence:
//     z_t = sigmoid(Wz.x_t + Uz.h_{t-1} + bz)      "update gate"
//     r_t = sigmoid(Wr.x_t + Ur.h_{t-1} + br)      "reset gate"
//     h~_t = tanh(Wh.x_t + Uh.(r_t * h_{t-1}) + bh) "candidate state"
//     h_t = (1 - z_t)*h_{t-1} + z_t*h~_t            "new state"
//     P(next token) = softmax(Wout.h_t + bout)
//
// Training truncates BPTT to fixed-length chunks (--seq-len) for
// tractability, and resets h to 0 at the start of every chunk rather than
// carrying it across the whole corpus — that keeps chunks independent, so
// the same "compute gradients in parallel, apply once" trick from v2 still
// works with zero unsafe code. At GENERATION time there's no such limit:
// the hidden state just keeps accumulating for as long as you generate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A genuinely tiny word-level language model with a GRU memory, trained via real backprop.
#[derive(Parser, Debug)]
#[command(name = "tinylm", about, long_about = None)]
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
    #[arg(long, default_value_t = 8000)]
    vocab_size: usize,

    /// Embedding dimension (size of each token's learned vector)
    #[arg(long, default_value_t = 16)]
    embed_dim: usize,

    /// GRU hidden state size
    #[arg(long, default_value_t = 96)]
    hidden_dim: usize,

    /// Truncated-BPTT chunk length: how many steps gradients actually flow
    /// back through. The hidden state itself only ever sees this much
    /// continuous history during training (it resets to 0 at each chunk
    /// boundary) — bigger catches longer-range dependencies but costs more
    /// per example.
    #[arg(long, default_value_t = 25)]
    seq_len: usize,

    /// Sequences per mini-batch. Each batch's gradients are computed in
    /// parallel across cores (one sequence's full BPTT per unit of work),
    /// then applied as one averaged update.
    #[arg(short = 'b', long, default_value_t = 512)]
    batch_size: usize,

    /// Load a previously trained model checkpoint instead of starting from
    /// random weights. Its vocab/embed_dim/hidden_dim win over the CLI
    /// flags above (they're baked into the checkpoint).
    #[arg(long, value_name = "PATH")]
    load_model: Option<PathBuf>,

    /// Save the trained model to this path after training (or immediately,
    /// if --load-model was given and --epochs is 0).
    #[arg(long, value_name = "PATH")]
    save_model: Option<PathBuf>,

    /// Also save (overwriting the same --save-model path) every N epochs
    /// during training, not just at the end. Real crash-safety for long
    /// runs: kill the process at epoch 40 of 83 and you still have epoch
    /// 40's weights on disk, not nothing.
    #[arg(long, value_name = "N")]
    checkpoint_every: Option<usize>,

    /// Don't train — measure real gradient-computation throughput on one
    /// actual batch (your data, your hardware, your hyperparameters), then
    /// print a suggested --epochs and an estimated total training time.
    /// Replaces hand-fit constants from someone else's run with a live
    /// measurement from this one.
    #[arg(long)]
    analyze: bool,

    /// Seed generation with this text: the model "reads" it (advancing its
    /// hidden state one token at a time, same math as training, just
    /// without a loss) before generating anything new. Echoed in the
    /// output, so unknown words showing up as <unk> is visible, not
    /// silent. Omit to start from a single (invisible) newline, as before.
    #[arg(long)]
    prompt: Option<String>,

    /// Seed the RNG used when sampling generated tokens, for reproducible
    /// output. The forward pass itself is a pure function of the weights —
    /// sampling is the *only* randomness in generation, so a fixed seed
    /// makes a run fully deterministic. Each of the 3 printed samples
    /// still differs from the others (seed, seed+1, seed+2), but rerunning
    /// with the same --seed reproduces that exact set again. Omit for
    /// fresh randomness every run (the default).
    #[arg(long)]
    seed: Option<u64>,

    /// Don't train or generate — run the (trained or loaded) model forward
    /// over real data and report the actual distribution of gate
    /// pre-activation values (what goes INTO sigmoid, not what comes out),
    /// so "how close to saturated is this model" is a measurement instead
    /// of a guess.
    #[arg(long)]
    diagnose_saturation: bool,

    /// Sampling temperature. <1 sharpens the distribution toward already-
    /// likely tokens (more repetitive, less erratic); >1 flattens it
    /// (more diverse, more erratic). 1.0 = the model's raw distribution.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Restrict sampling to only the K most probable tokens each step
    /// (renormalized). Cuts off the long tail of barely-likely tokens
    /// that's a lot of what makes small-model output read as noise.
    #[arg(long)]
    top_k: Option<usize>,

    /// Nucleus sampling: restrict to the smallest set of most-probable
    /// tokens whose cumulative probability reaches this threshold (e.g.
    /// 0.9), renormalized. An adaptive alternative/complement to --top-k.
    #[arg(long)]
    top_p: Option<f32>,
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

// ---------- Small dense-matrix helpers ----------
// Every weight matrix here is stored [in_dim * out_dim], row-major over the
// INPUT index — i.e. row `k` (a contiguous slice of length out_dim) holds
// all the weights fanning out of input unit k. That layout is what keeps
// every loop below contiguous (cache-friendly, auto-vectorizable) instead
// of striding through memory.
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

// out[h] += sum_k x[k] * mat[k*out_len + h]      (plain matrix-vector product)
fn linear_forward(mat: &[f32], x: &[f32], out: &mut [f32]) {
    let out_len = out.len();
    for (k, &xv) in x.iter().enumerate() {
        let row = &mat[k * out_len..(k + 1) * out_len];
        for h in 0..out_len {
            out[h] += xv * row[h];
        }
    }
}

// d_x[k] += sum_h mat[k*out_len + h] * d_out[h]  (product with the transpose)
fn linear_backward_input(mat: &[f32], d_out: &[f32], d_x: &mut [f32]) {
    let out_len = d_out.len();
    for k in 0..d_x.len() {
        let row = &mat[k * out_len..(k + 1) * out_len];
        let mut sum = 0.0f32;
        for h in 0..out_len {
            sum += row[h] * d_out[h];
        }
        d_x[k] += sum;
    }
}

// d_mat[k*out_len + h] += x[k] * d_out[h]         (outer product)
fn linear_backward_weight(d_mat: &mut [f32], x: &[f32], d_out: &[f32]) {
    let out_len = d_out.len();
    for (k, &xv) in x.iter().enumerate() {
        let row = &mut d_mat[k * out_len..(k + 1) * out_len];
        for h in 0..out_len {
            row[h] += xv * d_out[h];
        }
    }
}

// ---------- Model parameters ----------
struct Model {
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,

    embedding: Vec<f32>, // [vocab_size * embed_dim]

    // Gate weights, input-to-hidden: [embed_dim * hidden_dim] each.
    w_z: Vec<f32>,
    w_r: Vec<f32>,
    w_h: Vec<f32>,
    // Gate weights, hidden-to-hidden: [hidden_dim * hidden_dim] each.
    u_z: Vec<f32>,
    u_r: Vec<f32>,
    u_h: Vec<f32>,
    // Gate biases: [hidden_dim] each.
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_h: Vec<f32>,

    w_out: Vec<f32>, // [hidden_dim * vocab_size]
    b_out: Vec<f32>, // [vocab_size]
}

// One GRU step's outputs. Generation only needs `h` and `probs`; training
// keeps the whole thing around per-timestep so backprop-through-time has
// what it needs.
struct StepOut {
    z: Vec<f32>,
    r: Vec<f32>,
    hcand: Vec<f32>,
    h: Vec<f32>,
    probs: Vec<f32>,
}

// Gradients accumulated over a batch (or a chunk of one) — same shapes as
// Model's fields, plus the summed loss for reporting.
struct Grad {
    d_embedding: Vec<f32>,
    d_w_z: Vec<f32>,
    d_w_r: Vec<f32>,
    d_w_h: Vec<f32>,
    d_u_z: Vec<f32>,
    d_u_r: Vec<f32>,
    d_u_h: Vec<f32>,
    d_b_z: Vec<f32>,
    d_b_r: Vec<f32>,
    d_b_h: Vec<f32>,
    d_w_out: Vec<f32>,
    d_b_out: Vec<f32>,
    loss: f32,
}

impl Grad {
    fn zeros(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        Grad {
            d_embedding: vec![0.0; vocab_size * embed_dim],
            d_w_z: vec![0.0; embed_dim * hidden_dim],
            d_w_r: vec![0.0; embed_dim * hidden_dim],
            d_w_h: vec![0.0; embed_dim * hidden_dim],
            d_u_z: vec![0.0; hidden_dim * hidden_dim],
            d_u_r: vec![0.0; hidden_dim * hidden_dim],
            d_u_h: vec![0.0; hidden_dim * hidden_dim],
            d_b_z: vec![0.0; hidden_dim],
            d_b_r: vec![0.0; hidden_dim],
            d_b_h: vec![0.0; hidden_dim],
            d_w_out: vec![0.0; hidden_dim * vocab_size],
            d_b_out: vec![0.0; vocab_size],
            loss: 0.0,
        }
    }

    fn add_assign(&mut self, other: &Grad) {
        macro_rules! add_all {
            ($($field:ident),+) => {
                $(for (a, b) in self.$field.iter_mut().zip(&other.$field) { *a += b; })+
            };
        }
        add_all!(d_embedding, d_w_z, d_w_r, d_w_h, d_u_z, d_u_r, d_u_h, d_b_z, d_b_r, d_b_h, d_w_out, d_b_out);
        self.loss += other.loss;
    }
}

// A trained model plus everything needed to make its token ids meaningful
// again later — this whole struct is exactly what "saving the model" means.
// Note there's no seq_len here: that's a training-time BPTT truncation
// length, not a property of the model itself — unlike v2's fixed context
// window, a GRU works over sequences of any length at inference time.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    vocab: Vec<String>,
    embed_dim: usize,
    hidden_dim: usize,
    embedding: Vec<f32>,
    w_z: Vec<f32>,
    w_r: Vec<f32>,
    w_h: Vec<f32>,
    u_z: Vec<f32>,
    u_r: Vec<f32>,
    u_h: Vec<f32>,
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_h: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
}

impl Model {
    fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier-ish init: scale by 1/sqrt(fan_in) per matrix so the
        // sigmoid/tanh gates start in their responsive range.
        let mut rand_vec = |len: usize, fan_in: usize| -> Vec<f32> {
            let scale = 1.0 / (fan_in as f32).sqrt();
            (0..len).map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale).collect()
        };
        Model {
            vocab_size,
            embed_dim,
            hidden_dim,
            embedding: rand_vec(vocab_size * embed_dim, embed_dim),
            w_z: rand_vec(embed_dim * hidden_dim, embed_dim),
            w_r: rand_vec(embed_dim * hidden_dim, embed_dim),
            w_h: rand_vec(embed_dim * hidden_dim, embed_dim),
            u_z: rand_vec(hidden_dim * hidden_dim, hidden_dim),
            u_r: rand_vec(hidden_dim * hidden_dim, hidden_dim),
            u_h: rand_vec(hidden_dim * hidden_dim, hidden_dim),
            b_z: vec![0.0; hidden_dim],
            b_r: vec![0.0; hidden_dim],
            b_h: vec![0.0; hidden_dim],
            w_out: rand_vec(hidden_dim * vocab_size, hidden_dim),
            b_out: vec![0.0; vocab_size],
        }
    }

    fn from_checkpoint(ckpt: Checkpoint) -> (Self, Vec<String>) {
        let vocab_size = ckpt.vocab.len();
        let model = Model {
            vocab_size,
            embed_dim: ckpt.embed_dim,
            hidden_dim: ckpt.hidden_dim,
            embedding: ckpt.embedding,
            w_z: ckpt.w_z,
            w_r: ckpt.w_r,
            w_h: ckpt.w_h,
            u_z: ckpt.u_z,
            u_r: ckpt.u_r,
            u_h: ckpt.u_h,
            b_z: ckpt.b_z,
            b_r: ckpt.b_r,
            b_h: ckpt.b_h,
            w_out: ckpt.w_out,
            b_out: ckpt.b_out,
        };
        (model, ckpt.vocab)
    }

    fn to_checkpoint(&self, vocab: &[String]) -> Checkpoint {
        Checkpoint {
            vocab: vocab.to_vec(),
            embed_dim: self.embed_dim,
            hidden_dim: self.hidden_dim,
            embedding: self.embedding.clone(),
            w_z: self.w_z.clone(),
            w_r: self.w_r.clone(),
            w_h: self.w_h.clone(),
            u_z: self.u_z.clone(),
            u_r: self.u_r.clone(),
            u_h: self.u_h.clone(),
            b_z: self.b_z.clone(),
            b_r: self.b_r.clone(),
            b_h: self.b_h.clone(),
            w_out: self.w_out.clone(),
            b_out: self.b_out.clone(),
        }
    }

    // ---------- One GRU step (forward) ----------
    // Takes the current token and the previous hidden state, returns the
    // new hidden state and the output distribution over the next token.
    // Used both by generation (which only keeps `h`/`probs`) and by
    // training (which keeps everything, for backprop-through-time).
    fn step(&self, tok: usize, h_prev: &[f32]) -> StepOut {
        let embed_dim = self.embed_dim;
        let hidden_dim = self.hidden_dim;
        let vocab_size = self.vocab_size;
        let x = &self.embedding[tok * embed_dim..tok * embed_dim + embed_dim];

        let mut az = self.b_z.clone();
        linear_forward(&self.w_z, x, &mut az);
        linear_forward(&self.u_z, h_prev, &mut az);
        let z: Vec<f32> = az.iter().map(|&v| sigmoid(v)).collect();

        let mut ar = self.b_r.clone();
        linear_forward(&self.w_r, x, &mut ar);
        linear_forward(&self.u_r, h_prev, &mut ar);
        let r: Vec<f32> = ar.iter().map(|&v| sigmoid(v)).collect();

        let rh: Vec<f32> = (0..hidden_dim).map(|i| r[i] * h_prev[i]).collect();
        let mut ah = self.b_h.clone();
        linear_forward(&self.w_h, x, &mut ah);
        linear_forward(&self.u_h, &rh, &mut ah);
        let hcand: Vec<f32> = ah.iter().map(|&v| v.tanh()).collect();

        let h: Vec<f32> = (0..hidden_dim).map(|i| (1.0 - z[i]) * h_prev[i] + z[i] * hcand[i]).collect();

        let mut logits = self.b_out.clone();
        linear_forward(&self.w_out, &h, &mut logits);
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();
        let _ = vocab_size;

        StepOut { z, r, hcand, h, probs }
    }

    // Diagnostic only: recompute just the two gates' pre-activation values
    // (what feeds INTO sigmoid, before it gets squashed) without running
    // the rest of a step. Not used by training or generation — exists so
    // --diagnose-saturation can report real numbers instead of a guess.
    fn gate_preactivations(&self, tok: usize, h_prev: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let embed_dim = self.embed_dim;
        let x = &self.embedding[tok * embed_dim..tok * embed_dim + embed_dim];

        let mut az = self.b_z.clone();
        linear_forward(&self.w_z, x, &mut az);
        linear_forward(&self.u_z, h_prev, &mut az);

        let mut ar = self.b_r.clone();
        linear_forward(&self.w_r, x, &mut ar);
        linear_forward(&self.u_r, h_prev, &mut ar);

        (az, ar)
    }

    // ---------- Loss + backprop-through-time ----------
    // Runs `step` forward across the whole chunk, caching every timestep's
    // intermediates, then walks backward from the last step to the first,
    // carrying a `d_h_next` gradient — the standard BPTT recursion: each
    // step's total dh is "gradient from its own output" plus "gradient
    // that flowed back from being used as the next step's h_prev".
    //
    // Purely read-only w.r.t. `self` — safe to call from many threads at
    // once, each accumulating into its own `Grad`. That split (read-only
    // parallel gradient computation, then one serial update) is what lets
    // mini-batches parallelize without any unsafe code or locking.
    fn accumulate_gradients(&self, starts: &[usize], ids: &[usize], seq_len: usize) -> Grad {
        let embed_dim = self.embed_dim;
        let hidden_dim = self.hidden_dim;
        let vocab_size = self.vocab_size;
        let mut grad = Grad::zeros(vocab_size, embed_dim, hidden_dim);

        for &start in starts {
            let seq = &ids[start..start + seq_len + 1]; // seq_len inputs + 1 final target

            // ---- forward, caching every step ----
            let mut steps: Vec<StepOut> = Vec::with_capacity(seq_len);
            let mut h_prevs: Vec<Vec<f32>> = Vec::with_capacity(seq_len); // h_{t-1} fed into step t
            let mut h_prev = vec![0.0f32; hidden_dim]; // h_0 = 0: each chunk starts with no memory
            for t in 0..seq_len {
                h_prevs.push(h_prev.clone());
                let out = self.step(seq[t], &h_prev);
                let target = seq[t + 1];
                grad.loss += -(out.probs[target] + 1e-9).ln();
                h_prev = out.h.clone();
                steps.push(out);
            }

            // ---- backward through time ----
            let mut d_h_next = vec![0.0f32; hidden_dim]; // gradient arriving from step t+1
            for t in (0..seq_len).rev() {
                let tok = seq[t];
                let target = seq[t + 1];
                let x = &self.embedding[tok * embed_dim..tok * embed_dim + embed_dim];
                let h_prev = &h_prevs[t];
                let out = &steps[t];

                // Output layer: cross-entropy/softmax gradient, same clean
                // formula as always: dLoss/dLogit[j] = probs[j] - 1{j==target}.
                let mut d_logits = out.probs.clone();
                d_logits[target] -= 1.0;
                linear_backward_weight(&mut grad.d_w_out, &out.h, &d_logits);
                for j in 0..vocab_size {
                    grad.d_b_out[j] += d_logits[j];
                }

                // Total gradient w.r.t. h_t: from this step's own output,
                // plus whatever flowed back from step t+1 using h_t as its
                // h_prev.
                let mut d_h = vec![0.0f32; hidden_dim];
                linear_backward_input(&self.w_out, &d_logits, &mut d_h);
                for i in 0..hidden_dim {
                    d_h[i] += d_h_next[i];
                }

                // h_t = (1-z)*h_prev + z*hcand
                let mut d_hcand = vec![0.0f32; hidden_dim];
                let mut d_z = vec![0.0f32; hidden_dim];
                let mut d_h_prev = vec![0.0f32; hidden_dim]; // accumulates all contributions to d(h_{t-1})
                for i in 0..hidden_dim {
                    d_hcand[i] = d_h[i] * out.z[i];
                    d_z[i] = d_h[i] * (out.hcand[i] - h_prev[i]);
                    d_h_prev[i] = d_h[i] * (1.0 - out.z[i]);
                }

                // Candidate: h~ = tanh(Wh.x + Uh.(r*h_prev) + bh)
                let d_ah: Vec<f32> = (0..hidden_dim).map(|i| d_hcand[i] * (1.0 - out.hcand[i] * out.hcand[i])).collect();
                linear_backward_weight(&mut grad.d_w_h, x, &d_ah);
                for i in 0..hidden_dim {
                    grad.d_b_h[i] += d_ah[i];
                }
                let rh: Vec<f32> = (0..hidden_dim).map(|i| out.r[i] * h_prev[i]).collect();
                linear_backward_weight(&mut grad.d_u_h, &rh, &d_ah);
                let mut d_x = vec![0.0f32; embed_dim];
                linear_backward_input(&self.w_h, &d_ah, &mut d_x);
                let mut d_rh = vec![0.0f32; hidden_dim];
                linear_backward_input(&self.u_h, &d_ah, &mut d_rh);
                let mut d_r = vec![0.0f32; hidden_dim];
                for i in 0..hidden_dim {
                    d_r[i] = d_rh[i] * h_prev[i];
                    d_h_prev[i] += d_rh[i] * out.r[i];
                }

                // Reset gate: r = sigmoid(Wr.x + Ur.h_prev + br)
                let d_ar: Vec<f32> = (0..hidden_dim).map(|i| d_r[i] * out.r[i] * (1.0 - out.r[i])).collect();
                linear_backward_weight(&mut grad.d_w_r, x, &d_ar);
                for i in 0..hidden_dim {
                    grad.d_b_r[i] += d_ar[i];
                }
                linear_backward_weight(&mut grad.d_u_r, h_prev, &d_ar);
                linear_backward_input(&self.w_r, &d_ar, &mut d_x);
                linear_backward_input(&self.u_r, &d_ar, &mut d_h_prev);

                // Update gate: z = sigmoid(Wz.x + Uz.h_prev + bz)
                let d_az: Vec<f32> = (0..hidden_dim).map(|i| d_z[i] * out.z[i] * (1.0 - out.z[i])).collect();
                linear_backward_weight(&mut grad.d_w_z, x, &d_az);
                for i in 0..hidden_dim {
                    grad.d_b_z[i] += d_az[i];
                }
                linear_backward_weight(&mut grad.d_u_z, h_prev, &d_az);
                linear_backward_input(&self.w_z, &d_az, &mut d_x);
                linear_backward_input(&self.u_z, &d_az, &mut d_h_prev);

                // Scatter this step's input gradient into the embedding
                // row for whichever token it actually was — uses +=, so a
                // token appearing at multiple timesteps/sequences in this
                // chunk correctly accumulates gradient from every occurrence.
                let dst = &mut grad.d_embedding[tok * embed_dim..tok * embed_dim + embed_dim];
                for i in 0..embed_dim {
                    dst[i] += d_x[i];
                }

                d_h_next = d_h_prev; // carry into step t-1
            }
        }

        grad
    }

    // Apply one averaged update from a batch's accumulated gradients —
    // same "move opposite the gradient" rule as plain SGD, amortized over
    // batch_len examples instead of one.
    fn apply_gradients(&mut self, grad: &Grad, learning_rate: f32, batch_len: usize) {
        let scale = learning_rate / batch_len as f32;
        macro_rules! apply_all {
            ($(($field:ident, $dfield:ident)),+) => {
                $(for (p, g) in self.$field.iter_mut().zip(&grad.$dfield) { *p -= scale * g; })+
            };
        }
        apply_all!(
            (embedding, d_embedding),
            (w_z, d_w_z),
            (w_r, d_w_r),
            (w_h, d_w_h),
            (u_z, d_u_z),
            (u_r, d_u_r),
            (u_h, d_u_h),
            (b_z, d_b_z),
            (b_r, d_b_r),
            (b_h, d_b_h),
            (w_out, d_w_out),
            (b_out, d_b_out)
        );
    }
}

#[derive(Clone, Copy)]
struct SamplingConfig {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

// Reshape a probability distribution before sampling from it. Doesn't
// touch the model or its output layer at all — this is purely about
// which of the model's own predictions we're willing to consider.
//
// Temperature: raising each probability to the power 1/T and
// renormalizing is mathematically identical to softmax(logits/T) — same
// result, without needing the raw logits. T<1 sharpens the distribution
// toward already-likely tokens; T>1 flattens it toward uniform.
//
// top-k / top-p (nucleus): zero out everything outside the kept set and
// renormalize what's left. Both exist to cut off the long tail of
// barely-likely tokens — sampling from the FULL distribution every step
// means occasionally drawing from that tail, which is a lot of what
// makes small-model output read as noise instead of "weird but
// purposeful."
fn reshape_for_sampling(probs: &[f32], cfg: SamplingConfig) -> Vec<f32> {
    let mut p: Vec<f32> = if (cfg.temperature - 1.0).abs() < 1e-6 {
        probs.to_vec()
    } else {
        let scaled: Vec<f32> = probs.iter().map(|&x| x.max(1e-12).powf(1.0 / cfg.temperature)).collect();
        let sum: f32 = scaled.iter().sum();
        scaled.iter().map(|&x| x / sum).collect()
    };

    let renormalize = |p: &mut [f32]| {
        let sum: f32 = p.iter().sum();
        if sum > 0.0 {
            for x in p.iter_mut() {
                *x /= sum;
            }
        }
    };

    if let Some(k) = cfg.top_k {
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_unstable_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        for &i in idx.iter().skip(k) {
            p[i] = 0.0;
        }
        renormalize(&mut p);
    }

    if let Some(top_p) = cfg.top_p {
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_unstable_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            cum += p[i];
            if cum >= top_p {
                cutoff = rank + 1;
                break;
            }
        }
        for &i in idx.iter().skip(cutoff) {
            p[i] = 0.0;
        }
        renormalize(&mut p);
    }

    p
}

// ---------- Generation (autoregressive: feed output back in as next input) ----------
// No fixed window here — the hidden state just keeps accumulating for as
// long as we keep generating, which is the whole point of an RNN over a
// fixed-context model.
fn generate(
    model: &Model,
    vocab: &[String],
    token_to_id: &HashMap<String, usize>,
    prompt_ids: &[usize],
    length: usize,
    seed: Option<u64>,
    sampling: SamplingConfig,
) -> String {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    // Seed sequence: the caller's prompt (tokenized against this model's
    // own vocab — unknown words become <unk>), or a single newline
    // ("start of a line") if no prompt was given.
    let newline_id = token_to_id.get("\n").copied().unwrap_or(0);
    let seed_ids: Vec<usize> = if prompt_ids.is_empty() { vec![newline_id] } else { prompt_ids.to_vec() };

    let mut h = vec![0.0f32; model.hidden_dim];
    let mut out_tokens: Vec<String> = Vec::with_capacity(seed_ids.len() + length);
    let mut last_probs = vec![0.0f32; model.vocab_size];

    // "Read" the seed: advance the hidden state through it token by token.
    // This is mechanically exactly what prompting is — the model's state
    // after the last prompt token IS its distribution over what comes
    // next, the same math as training, just with no target/loss.
    for &id in &seed_ids {
        out_tokens.push(vocab[id].clone());
        let out = model.step(id, &h);
        h = out.h;
        last_probs = out.probs;
    }

    for _ in 0..length {
        // Sample rather than always taking the top choice — this is the
        // actual, sole source of "not the same twice." The forward pass
        // is a pure function of the weights; only this draw is random,
        // and only when `seed` is None. `sampling` narrows/reshapes the
        // distribution first (temperature/top-k/top-p) but never adds
        // any information the model didn't already produce.
        let sampling_probs = reshape_for_sampling(&last_probs, sampling);
        let mut r: f32 = rng.random();
        let mut chosen_id = 0;
        for (j, &p) in sampling_probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                chosen_id = j;
                break;
            }
        }

        out_tokens.push(vocab[chosen_id].clone());
        let out = model.step(chosen_id, &h);
        h = out.h;
        last_probs = out.probs; // <-- the model's own output becomes its next input
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
            "Loaded model from {} (vocab {}, embed_dim {}, hidden_dim {})",
            load_path.display(),
            ckpt.vocab.len(),
            ckpt.embed_dim,
            ckpt.hidden_dim
        );
        let (model, vocab) = Model::from_checkpoint(ckpt);
        let token_to_id: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        (vocab, token_to_id, model)
    } else {
        let (vocab, token_to_id) = build_vocab(&tokens_all, cli.vocab_size);
        let model = Model::new(vocab.len(), cli.embed_dim, cli.hidden_dim);
        (vocab, token_to_id, model)
    };
    println!("Vocab size: {} (includes <unk>)", vocab.len());

    // ---------- 3. Build training chunks ----------
    // Non-overlapping windows of seq_len+1 tokens (seq_len inputs, shifted
    // by one for targets). Each chunk is trained independently — h resets
    // to 0 at every chunk start (see the big comment at the top of the file
    // for why that's the right tradeoff for parallelizing this safely).
    let seq_len = cli.seq_len.max(1);
    let ids: Vec<usize> = tokens_all.iter().map(|t| token_to_id.get(t.as_str()).copied().unwrap_or(0)).collect();

    let mut chunk_starts: Vec<usize> = Vec::new();
    let mut pos = 0;
    while pos + seq_len + 1 <= ids.len() {
        chunk_starts.push(pos);
        pos += seq_len;
    }
    let num_examples = chunk_starts.len();

    // ---------- 3b. --analyze: measure, don't train ----------
    // Runs a handful of REAL batches through the exact same parallel
    // gradient path training uses, on this machine, with these
    // hyperparameters — then extrapolates. No baked-in constants from
    // someone else's run; this is a live measurement of your run.
    if cli.analyze {
        if num_examples == 0 {
            println!("Not enough tokens ({}) for even one sequence of length {seq_len} — nothing to analyze.", ids.len());
            return;
        }
        let batch_size = cli.batch_size.max(1).min(num_examples);
        let num_threads = rayon::current_num_threads();
        let vocab_size = model.vocab_size;
        let embed_dim = model.embed_dim;
        let hidden_dim = model.hidden_dim;
        let batches_per_epoch = num_examples.div_ceil(batch_size);
        let chunk_len = batch_size.div_ceil(num_threads).max(1);

        let run_batch = |starts: &[usize]| {
            starts
                .par_chunks(chunk_len)
                .map(|chunk| model.accumulate_gradients(chunk, &ids, seq_len))
                .reduce(
                    || Grad::zeros(vocab_size, embed_dim, hidden_dim),
                    |mut a, b| {
                        a.add_assign(&b);
                        a
                    },
                )
        };

        run_batch(&chunk_starts[0..batch_size]); // warm up the thread pool/caches, discard timing

        let sample_count = batches_per_epoch.min(4);
        let mut total = std::time::Duration::ZERO;
        for i in 0..sample_count {
            let start_idx = (i * batch_size).min(num_examples - batch_size);
            let t0 = Instant::now();
            run_batch(&chunk_starts[start_idx..start_idx + batch_size]);
            total += t0.elapsed();
        }
        let avg_batch_time = total / sample_count as u32;
        let epoch_time = avg_batch_time * batches_per_epoch as u32;

        let total_tokens = tokens_all.len();
        // Floor scales with vocab_size, not just a flat 15. Why: a bigger
        // output softmax has more classes to calibrate, which takes more
        // EPOCHS (more full passes through the LR schedule) to converge —
        // not just more raw token exposure. Measured directly: sherlock
        // (vocab 8000) got MORE total token-exposures than mothergoose
        // (vocab 3179, 83 epochs) by hitting the old flat floor of 15
        // epochs, yet converged to a much smaller fraction of its
        // distance-from-random-baseline (~5% vs ~15%) — proof it was
        // epochs, not tokens, that were the binding constraint. Divisor
        // calibrated off that comparison (vocab_size / 200 ~= 40 for
        // vocab 8000, in the range that comparison suggested was needed).
        let vocab_floor = (vocab.len() as f64 / 200.0).round() as usize;
        let min_epochs = 15usize.max(vocab_floor);
        let suggested_epochs = ((2_000_000f64 / total_tokens.max(1) as f64).round() as usize).clamp(min_epochs, 400);
        let total_time = epoch_time * suggested_epochs as u32;

        println!("\n--- Analysis ({} sample batch{}, not trained) ---", sample_count, if sample_count == 1 { "" } else { "es" });
        println!("Tokens: {total_tokens}   Vocab: {}   Sequences: {num_examples} (seq-len {seq_len})", vocab.len());
        println!("Measured: {avg_batch_time:.2?}/batch, {batches_per_epoch} batches/epoch -> ~{epoch_time:.2?}/epoch");
        println!("Suggested epochs (~2M token-exposure budget, floor {min_epochs} [scales with vocab] / cap 400): {suggested_epochs}");
        println!("Estimated total training time: ~{total_time:.2?}");
        println!("\nSuggested command:");
        println!(
            "  tinylm --data-set {} --epochs {suggested_epochs} --save-model <path>",
            data_path.display()
        );
        return;
    }

    // ---------- 4. Training loop (mini-batch, parallel across cores) ----------
    if cli.epochs > 0 && num_examples > 0 {
        let batch_size = cli.batch_size.max(1);
        let num_threads = rayon::current_num_threads();
        let vocab_size = model.vocab_size;
        let embed_dim = model.embed_dim;
        let hidden_dim = model.hidden_dim;

        println!(
            "Training on {} sequences of {} tokens each ({} tokens/batch avg), batch size {}, {} threads available",
            num_examples,
            seq_len,
            batch_size * seq_len,
            batch_size,
            num_threads
        );

        let mut order: Vec<usize> = (0..num_examples).collect();
        let mut rng = rand::rng();

        let start_time = Instant::now();
        for epoch in 0..cli.epochs {
            // Cosine decay from cli.lr down to 10% of it over the course of
            // training: full step size early (when big moves toward a
            // better region are cheap and useful), progressively smaller
            // steps later (when you're closer to a minimum and a big step
            // would just overshoot it). Never decays all the way to zero,
            // so the last epoch still actually learns something.
            let progress = epoch as f32 / cli.epochs.max(1) as f32;
            let lr = cli.lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()));

            order.shuffle(&mut rng); // real training shuffles each epoch; so do we
            let mut total_loss = 0.0f32;

            for batch_start in (0..order.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(order.len());
                let batch_starts: Vec<usize> = order[batch_start..batch_end].iter().map(|&i| chunk_starts[i]).collect();

                let chunk_len = batch_starts.len().div_ceil(num_threads).max(1);
                let grad = batch_starts
                    .par_chunks(chunk_len)
                    .map(|chunk| model.accumulate_gradients(chunk, &ids, seq_len))
                    .reduce(
                        || Grad::zeros(vocab_size, embed_dim, hidden_dim),
                        |mut a, b| {
                            a.add_assign(&b);
                            a
                        },
                    );

                total_loss += grad.loss;
                model.apply_gradients(&grad, lr, batch_starts.len() * seq_len);
            }

            let report_every = (cli.epochs / 20).max(1);
            if epoch % report_every == 0 || epoch == cli.epochs - 1 {
                println!("epoch {}: avg loss = {:.4} (lr={:.4})", epoch, total_loss / (num_examples * seq_len) as f32, lr);
            }

            if let (Some(n), Some(save_path)) = (cli.checkpoint_every, &cli.save_model) {
                if (epoch + 1) % n == 0 {
                    save_checkpoint(save_path, &model, &vocab).expect("failed to save mid-training checkpoint");
                    println!("  (checkpoint saved at epoch {epoch})");
                }
            }
        }
        println!("Training took {:.2?}", start_time.elapsed());
    } else if cli.epochs == 0 {
        println!("--epochs 0: skipping training, using the model as loaded.");
    }

    // ---------- 5. Save, if asked ----------
    if let Some(save_path) = &cli.save_model {
        save_checkpoint(save_path, &model, &vocab).expect("failed to save model checkpoint");
        println!("Saved model to {}", save_path.display());
    }

    // ---------- 5b. --diagnose-saturation: measure, don't generate ----------
    if cli.diagnose_saturation {
        let sample_len = ids.len().min(20_000);
        if sample_len == 0 {
            println!("No tokens to diagnose against.");
            return;
        }
        let mut h = vec![0.0f32; model.hidden_dim];
        let mut az_all: Vec<f32> = Vec::with_capacity(sample_len * model.hidden_dim);
        let mut ar_all: Vec<f32> = Vec::with_capacity(sample_len * model.hidden_dim);
        for &tok in &ids[..sample_len] {
            let (az, ar) = model.gate_preactivations(tok, &h);
            az_all.extend_from_slice(&az);
            ar_all.extend_from_slice(&ar);
            h = model.step(tok, &h).h; // advance state exactly as generation/inference would
        }

        fn report(name: &str, values: &[f32]) {
            let n = values.len() as f64;
            let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let pct_beyond = |t: f32| values.iter().filter(|&&v| v.abs() > t).count() as f64 / n * 100.0;
            println!(
                "{name}: n={} mean={mean:.3} std={std:.3} range=[{min:.3}, {max:.3}]\n  fraction with |x| beyond: >2 = {:.3}%   >4 = {:.3}%   >6 = {:.3}%",
                values.len(),
                pct_beyond(2.0),
                pct_beyond(4.0),
                pct_beyond(6.0)
            );
        }

        println!("\n--- Gate pre-activation distribution over {sample_len} real tokens ---");
        report("update gate (az, feeds sigmoid)", &az_all);
        report("reset gate  (ar, feeds sigmoid)", &ar_all);
        return;
    }

    // ---------- 6. Generation ----------
    let prompt_ids: Vec<usize> = match &cli.prompt {
        Some(text) => tokenize(text).iter().map(|t| token_to_id.get(t.as_str()).copied().unwrap_or(0)).collect(),
        None => Vec::new(),
    };

    let sampling = SamplingConfig { temperature: cli.temperature, top_k: cli.top_k, top_p: cli.top_p };

    println!("\n--- Generated text ---");
    for i in 0..3u64 {
        let sample_seed = cli.seed.map(|s| s.wrapping_add(i));
        println!(
            "{}\n---",
            generate(&model, &vocab, &token_to_id, &prompt_ids, cli.length, sample_seed, sampling)
        );
    }
}

// ---------- Correctness check: analytic gradients vs finite differences ----------
// The whole point of this file is "real backprop, not the finite-difference
// approximation" — so here, finite differences are used the other way
// around: as a cheap, mechanical proof that the hand-derived BPTT above is
// actually correct, on a model small enough to compute both by hand-ish.
// A perturb-and-compare check like this is the standard way to trust a new
// backward pass before spending real compute training it.
#[cfg(test)]
mod tests {
    use super::*;

    fn check_param(model: &mut Model, loss_fn: &dyn Fn(&Model) -> f32, get: impl Fn(&mut Model) -> &mut f32, analytic: f32, name: &str) {
        let eps = 1e-2f32;
        let orig = *get(model);
        *get(model) = orig + eps;
        let loss_plus = loss_fn(model);
        *get(model) = orig - eps;
        let loss_minus = loss_fn(model);
        *get(model) = orig;

        let numeric = (loss_plus - loss_minus) / (2.0 * eps);
        let diff = (numeric - analytic).abs();
        let rel = diff / numeric.abs().max(analytic.abs()).max(1e-6);
        println!("{name}: analytic={analytic:.6} numeric={numeric:.6} rel_err={rel:.6}");
        assert!(rel < 0.03 || diff < 1e-2, "{name} mismatch: analytic={analytic}, numeric={numeric}");
    }

    #[test]
    fn gru_gradients_match_finite_differences() {
        let vocab_size = 6;
        let embed_dim = 3;
        let hidden_dim = 4;
        let seq_len = 4;
        let mut model = Model::new(vocab_size, embed_dim, hidden_dim);
        let ids: Vec<usize> = vec![1, 2, 3, 4, 0]; // seq_len + 1 tokens
        let starts = vec![0usize];

        let grad = model.accumulate_gradients(&starts, &ids, seq_len);

        let loss_fn = |m: &Model| -> f32 {
            let mut total = 0.0f32;
            for &start in &starts {
                let seq = &ids[start..start + seq_len + 1];
                let mut h = vec![0.0f32; hidden_dim];
                for t in 0..seq_len {
                    let out = m.step(seq[t], &h);
                    total += -(out.probs[seq[t + 1]] + 1e-9).ln();
                    h = out.h;
                }
            }
            total
        };

        check_param(&mut model, &loss_fn, |m| &mut m.w_out[5], grad.d_w_out[5], "w_out[5]");
        check_param(&mut model, &loss_fn, |m| &mut m.b_out[2], grad.d_b_out[2], "b_out[2]");
        check_param(&mut model, &loss_fn, |m| &mut m.w_z[1], grad.d_w_z[1], "w_z[1]");
        check_param(&mut model, &loss_fn, |m| &mut m.u_z[2], grad.d_u_z[2], "u_z[2]");
        check_param(&mut model, &loss_fn, |m| &mut m.w_r[0], grad.d_w_r[0], "w_r[0]");
        check_param(&mut model, &loss_fn, |m| &mut m.u_r[3], grad.d_u_r[3], "u_r[3]");
        check_param(&mut model, &loss_fn, |m| &mut m.w_h[2], grad.d_w_h[2], "w_h[2]");
        check_param(&mut model, &loss_fn, |m| &mut m.u_h[1], grad.d_u_h[1], "u_h[1]");
        check_param(&mut model, &loss_fn, |m| &mut m.b_z[0], grad.d_b_z[0], "b_z[0]");
        check_param(&mut model, &loss_fn, |m| &mut m.b_r[1], grad.d_b_r[1], "b_r[1]");
        check_param(&mut model, &loss_fn, |m| &mut m.b_h[2], grad.d_b_h[2], "b_h[2]");
        // token 1 occurs at t=0, token 4 at t=3 — checks both an early and
        // a late timestep's contribution to the embedding gradient.
        check_param(&mut model, &loss_fn, |m| &mut m.embedding[1 * embed_dim], grad.d_embedding[1 * embed_dim], "embedding[tok1,0]");
        check_param(&mut model, &loss_fn, |m| &mut m.embedding[4 * embed_dim + 2], grad.d_embedding[4 * embed_dim + 2], "embedding[tok4,2]");
    }
}
