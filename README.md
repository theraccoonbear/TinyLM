# tiny_llm

A genuinely tiny language model, in one readable Rust file (`src/main.rs`) —
built to show the actual mechanics of how these things learn, not just call
an API. Real tokenization, real embeddings, real backprop, real
backprop-*through-time*. No autodiff library, no ML framework — every
gradient is hand-derived and hand-coded so you can read the whole training
loop and believe it.

It will never write a real sonnet. It's small on purpose. But it will
learn actual words, actual character names, and actual local phrasing from
whatever text you point it at, in well under a couple of minutes for
most-sized inputs.

## 1. What this does

You give it a text file. It:

1. Splits that text into tokens (words, punctuation, line breaks).
2. Trains a small recurrent neural network to predict "what token comes
   next" from everything it's seen so far in the current sequence.
3. Uses that trained model to generate new text, one token at a time,
   sampling from its own predictions and feeding its own output back in as
   the next input.

The output is not coherent prose. It's real words and real local
structure (names, common phrases, learned line breaks) recombined in
grammar-free order — "readable, often atrocious." That's the honest
ceiling for a model this size, and it's architectural, not a matter of
more training: see [How it does it](#2-how-it-does-it) for why.

You can train it on anything — we've run it on the complete works of
Shakespeare, Alice in Wonderland, Sherlock Holmes, Grimm's Fairy Tales,
and The Real Mother Goose. Pretrained models for all of these live in
[`models/`](models/).

## 2. How it does it

### Evolution of this project

This file has gone through three architectures, each fixing a real
limitation of the last:

1. **Character-level, no context.** Predicted the next *character* from
   only the current one — an order-1 Markov chain wearing a neural net
   costume. Could never produce real words no matter how long it trained,
   because it had no concept of "word" at all.
2. **Word-level, fixed context window.** Tokens became words. Prediction
   was conditioned on the last 4 tokens (a small [Bengio 2003][bengio]
   neural language model). Real words, finally — but zero memory beyond
   that fixed window.
3. **Word-level, recurrent (current).** A GRU (Gated Recurrent Unit)
   carries a hidden state across the *whole* sequence, updated at each
   step. This is the actual architectural lineage that led to modern
   LLMs: RNN → LSTM/GRU → Transformer. It's the first point in this
   project where "how much context can influence a prediction" isn't
   capped by a hardcoded window size.

[bengio]: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

### The pipeline, in order

**Tokenizer.** Runs of letters/apostrophes become one word token
(`thou'lt` stays whole). Every other punctuation character becomes its
own token. `\n` becomes its own token too, on purpose — it's the closest
thing to verse/paragraph structure the model can learn.

**Vocabulary.** The `N` most frequent tokens are kept (default 3000);
everything else collapses to `<unk>`. Real text vocabularies follow a
Zipf distribution — a small set of tokens covers most of the corpus — so
this bounds the (expensive) output layer without losing much coverage.

**The model**, per step `t`, given the current token embedded as `x_t`
and the previous hidden state `h_{t-1}`:

```
z_t     = sigmoid(Wz·x_t + Uz·h_{t-1} + bz)        update gate
r_t     = sigmoid(Wr·x_t + Ur·h_{t-1} + br)        reset gate
h~_t    = tanh(Wh·x_t + Uh·(r_t * h_{t-1}) + bh)   candidate state
h_t     = (1 - z_t)*h_{t-1} + z_t*h~_t             new hidden state
P(next) = softmax(Wout·h_t + bout)
```

**Training** is real backprop-through-time (BPTT): gradients are
hand-derived through every one of those equations, unrolled across a
truncated sequence chunk (`--seq-len`, default 25 tokens), then
propagated backward from the last step to the first. The hidden state
resets to zero at the start of every chunk — training never carries
memory across chunk boundaries — which keeps chunks independent and
enables the next point:

**Parallelism.** Gradient computation for a mini-batch is split across
CPU cores (via [rayon](https://github.com/rayon-rs/rayon)): each thread
does read-only forward+backward passes over its slice of the batch,
accumulating its own gradient totals. Those totals are summed and applied
as a *single* weight update, once, on the main thread. Because the
parallel phase never mutates shared state, this needs zero locks and zero
`unsafe` code.

**Correctness.** BPTT-by-hand is exactly the kind of code that's easy to
get subtly wrong. `src/main.rs` includes a `#[test]` that numerically
verifies every analytic gradient against finite differences (perturb a
weight by ±ε, compare `(loss(w+ε) - loss(w-ε)) / 2ε` to the hand-derived
gradient) — the same technique the very first version of this file used
*as its training method*, now repurposed as a correctness check on the
real thing. Run it with `cargo test -- --nocapture`.

**Generation** is autoregressive: feed in a token, get a probability
distribution over the next one, sample from it (not just take the top
choice — that's why output differs between runs), feed the sampled token
back in, repeat. Unlike training's truncated window, generation carries
the hidden state for as long as you keep generating — no length cap.

### Why it can't write real Shakespeare

This is a real architectural ceiling, not a training-time problem. A
model this size (default: 16-dim embeddings, 48-dim hidden state, 3000-
token vocabulary) has nowhere near the capacity to model English syntax —
it can pick up local statistical patterns (which words tend to follow
which, common short phrases, character names, where lines tend to break)
but has no representation of grammar, sentence structure, or meaning.
More epochs converge it closer to that ceiling faster; they don't raise
the ceiling. Raising the ceiling means a fundamentally bigger model
(more parameters, attention instead of a single recurrent state,
subword tokenization, orders of magnitude more data) — i.e., an actual
LLM instead of a tiny one.

## 3. How you train a model

```sh
cargo build --release
./target/release/tiny_llm --data-set path/to/your.txt --save-model my.model
```

That's the whole workflow: point it at a text file, tell it where to save
the result. Useful flags:

| Flag | Default | What it controls |
|---|---|---|
| `-d, --data-set <path>` | `training-data.txt` | Text file to train on |
| `-e, --epochs <n>` | 10 | Full passes over the data |
| `--lr <n>` | 0.1 | Learning rate (step size per update) |
| `--vocab-size <n>` | 3000 | Max distinct tokens; rest collapse to `<unk>` |
| `--embed-dim <n>` | 16 | Size of each token's learned embedding |
| `--hidden-dim <n>` | 48 | Size of the GRU's hidden state (memory capacity) |
| `--seq-len <n>` | 25 | Truncated-BPTT window: how far back gradients flow |
| `-b, --batch-size <n>` | 512 | Sequences per parallel mini-batch |
| `--max-chars <n>` | — | Truncate input to the first N characters |
| `--save-model <path>` | — | Where to write the trained checkpoint |
| `-l, --length <n>` | 80 | Tokens to generate per sample, after training |
| `--prompt <text>` | — | Seed generation with this text instead of a bare newline |
| `--seed <n>` | — | Seed the sampling RNG for reproducible output (omit for fresh randomness each run) |
| `--analyze` | off | Don't train — measure and estimate instead (see below) |

**Before committing to a long run**, use `--analyze`:

```sh
./target/release/tiny_llm --data-set corpora/shakespeare.txt --analyze
```

This runs a handful of *real* gradient-computation batches — your data,
your hyperparameters, this machine — through the exact same parallel code
path training uses, times them, and extrapolates: a suggested `--epochs`
(targeting a ~2M-token-exposure budget across all epochs, floored at 15
so huge corpora aren't starved, capped at 400 so tiny corpora don't run
forever), an estimated per-epoch and total training time, and a
ready-to-run command. It's a live measurement, not a canned formula — it
stays honest even if you change `--hidden-dim`, `--batch-size`, or run it
on different hardware.

## 4. How you use a model

Skip training entirely and generate from a saved checkpoint:

```sh
./target/release/tiny_llm --load-model models/macbeth.model --epochs 0 --length 200
```

`--load-model` restores the vocabulary and all learned weights — the
model's `embed-dim`/`hidden-dim` come from the checkpoint itself, not
your CLI flags, so you don't need to remember what you trained it with.
`--epochs 0` skips straight to generation. Omit it (or pass a positive
number) to keep training the loaded model further instead — checkpoints
are resumable.

Pretrained checkpoints in [`models/`](models/), one per corpus in
[`corpora/`](corpora/):

| Model | Trained on |
|---|---|
| `models/shakespeare.model` | Complete works of Shakespeare (boilerplate-stripped) |
| `models/macbeth.model` | Macbeth only |
| `models/alice.model` | Alice's Adventures in Wonderland |
| `models/sherlock.model` | The Adventures of Sherlock Holmes |
| `models/grimm.model` | Grimm's Fairy Tales |
| `models/mothergoose.model` | The Real Mother Goose (nursery rhymes) |
