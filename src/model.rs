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
// LLMs (RNN -> LSTM/GRU -> Transformer) — the first point in this project's
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

use rand::Rng;
use serde::{Deserialize, Serialize};

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
pub struct Model {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,

    pub embedding: Vec<f32>, // [vocab_size * embed_dim]

    // Gate weights, input-to-hidden: [embed_dim * hidden_dim] each.
    pub w_z: Vec<f32>,
    pub w_r: Vec<f32>,
    pub w_h: Vec<f32>,
    // Gate weights, hidden-to-hidden: [hidden_dim * hidden_dim] each.
    pub u_z: Vec<f32>,
    pub u_r: Vec<f32>,
    pub u_h: Vec<f32>,
    // Gate biases: [hidden_dim] each.
    pub b_z: Vec<f32>,
    pub b_r: Vec<f32>,
    pub b_h: Vec<f32>,

    pub w_out: Vec<f32>, // [hidden_dim * vocab_size]
    pub b_out: Vec<f32>, // [vocab_size]
}

// One GRU step's outputs. Generation only needs `h` and `probs`; training
// keeps the whole thing around per-timestep so backprop-through-time has
// what it needs.
pub struct StepOut {
    pub z: Vec<f32>,
    pub r: Vec<f32>,
    pub hcand: Vec<f32>,
    pub h: Vec<f32>,
    pub probs: Vec<f32>,
}

// Gradients accumulated over a batch (or a chunk of one) — same shapes as
// Model's fields, plus the summed loss for reporting.
pub struct Grad {
    pub d_embedding: Vec<f32>,
    pub d_w_z: Vec<f32>,
    pub d_w_r: Vec<f32>,
    pub d_w_h: Vec<f32>,
    pub d_u_z: Vec<f32>,
    pub d_u_r: Vec<f32>,
    pub d_u_h: Vec<f32>,
    pub d_b_z: Vec<f32>,
    pub d_b_r: Vec<f32>,
    pub d_b_h: Vec<f32>,
    pub d_w_out: Vec<f32>,
    pub d_b_out: Vec<f32>,
    pub loss: f32,
}

impl Grad {
    pub fn zeros(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
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

    pub fn add_assign(&mut self, other: &Grad) {
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
pub struct Checkpoint {
    pub vocab: Vec<String>,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub embedding: Vec<f32>,
    pub w_z: Vec<f32>,
    pub w_r: Vec<f32>,
    pub w_h: Vec<f32>,
    pub u_z: Vec<f32>,
    pub u_r: Vec<f32>,
    pub u_h: Vec<f32>,
    pub b_z: Vec<f32>,
    pub b_r: Vec<f32>,
    pub b_h: Vec<f32>,
    pub w_out: Vec<f32>,
    pub b_out: Vec<f32>,
}

impl Model {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
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

    pub fn from_checkpoint(ckpt: Checkpoint) -> (Self, Vec<String>) {
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

    pub fn to_checkpoint(&self, vocab: &[String]) -> Checkpoint {
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
    pub fn step(&self, tok: usize, h_prev: &[f32]) -> StepOut {
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
    pub fn gate_preactivations(&self, tok: usize, h_prev: &[f32]) -> (Vec<f32>, Vec<f32>) {
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
    pub fn accumulate_gradients(&self, starts: &[usize], ids: &[usize], seq_len: usize) -> Grad {
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
    pub fn apply_gradients(&mut self, grad: &Grad, learning_rate: f32, batch_len: usize) {
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

pub fn save_checkpoint(path: &std::path::Path, model: &Model, vocab: &[String]) -> std::io::Result<()> {
    let ckpt = model.to_checkpoint(vocab);
    let bytes = bincode::serialize(&ckpt).expect("failed to serialize model");
    std::fs::write(path, bytes)
}

pub fn load_checkpoint(path: &std::path::Path) -> Checkpoint {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read model checkpoint {}: {e}", path.display()));
    bincode::deserialize(&bytes).unwrap_or_else(|e| panic!("failed to parse model checkpoint {}: {e}", path.display()))
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
