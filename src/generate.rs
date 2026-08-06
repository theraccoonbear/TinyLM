// ---------- Generation (autoregressive: feed output back in as next input) ----------
// Everything here is inference-time only — nothing in this file trains
// anything. No fixed window either: the hidden state just keeps
// accumulating for as long as we keep generating, which is the whole
// point of an RNN over a fixed-context model.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::model::Model;
use crate::tokenizer::detokenize;

#[derive(Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
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

pub fn generate(
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
