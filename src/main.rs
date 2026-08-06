// TinyLM — a genuinely tiny language model, in readable Rust.
//
// This file is the CLI and the training/generation pipeline that drives
// everything else. The actual "how does it learn" content lives in three
// focused modules, each a self-contained chapter:
//
//   tokenizer.rs — text <-> tokens, and the vocabulary built from them
//   model.rs     — the GRU itself: forward pass, real backprop-through-time,
//                  checkpointing, and the gradient-check test that proves
//                  the hand-derived backward pass is correct
//   generate.rs  — autoregressive sampling from a trained model
//
// See model.rs's header for the actual architecture explanation (the GRU
// equations, why a hidden state instead of a fixed window, etc.) — this
// file is deliberately just plumbing: parse args, load data, call into
// the modules above, print the results.

mod generate;
mod model;
mod tokenizer;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use generate::{generate, SamplingConfig};
use model::{load_checkpoint, save_checkpoint, Grad, Model};
use tokenizer::{build_vocab, tokenize};

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
    // to 0 at every chunk start (see model.rs's header for why that's the
    // right tradeoff for parallelizing this safely).
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
