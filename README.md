# TinyLM

A genuinely tiny language model, in readable Rust — built to show the
actual mechanics of how these things learn, not just call an API. Real
tokenization, real embeddings, real backprop, real backprop-*through-time*.
No autodiff library, no ML framework — every gradient is hand-derived and
hand-coded so you can read the whole training loop and believe it.

The source is a few focused files, each a self-contained chapter:

| File | Lines | What's in it |
|---|---|---|
| [`tokenizer.rs`](src/tokenizer.rs) | ~90 | Text ↔ tokens, and the vocabulary built from them |
| [`model.rs`](src/model.rs) | ~550 | The GRU itself: forward pass, real backprop-through-time, checkpointing, and the gradient-check test that proves the hand-derived backward pass is correct |
| [`generate.rs`](src/generate.rs) | ~145 | Autoregressive sampling from a trained model |
| [`main.rs`](src/main.rs) | ~440 | CLI + orchestration — parses args, calls into the three files above, prints results |

`model.rs` is where the actual "how does it learn" story lives, including
the architecture write-up below. The other three are supporting cast.

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

This project has gone through three architectures, each fixing a real
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

**Vocabulary.** The `N` most frequent tokens are kept (default 8000);
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

Standard notation from the [GRU paper][gru]; matches the code's variable
names (`z`, `r`, `hcand`, `h`) directly. Per line:

- **`z_t`, update gate** — per memory slot, what fraction of new
  information should overwrite it (0–1, learned).
- **`r_t`, reset gate** — per slot, how much of the *old* memory to
  consider when proposing new content. What lets the model learn to
  deliberately forget.
- **`h~_t`, candidate** — the proposed new memory content, from the
  current input plus whatever old memory the reset gate let through.
- **`h_t`, new hidden state** — old memory and candidate, blended by the
  update gate. `z_t = 0` at a slot means "keep it exactly," so a slot can
  survive arbitrarily many steps if the model learns it should.
- **`P(next)`** — the current memory turned into next-token odds, same
  output layer as v2, just fed by a hidden state instead of a fixed
  window.

`sigmoid` squashes to (0, 1), so it's used for the gates (a percentage:
0 = closed, 1 = open). `tanh` squashes to (-1, 1), used for the
candidate since memory needs to move in either direction, not just
scale down. (`tanh` = hyperbolic tangent, unrelated to `arctan` despite
the shared letters; `tanh(x) = 2·sigmoid(2x) - 1`.)

[gru]: https://arxiv.org/abs/1406.1078

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
get subtly wrong. `model.rs` includes a `#[test]` that numerically
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
model this size (default: 16-dim embeddings, 96-dim hidden state, 8000-
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
./target/release/tinylm --data-set path/to/your.txt --save-model my.model
```

That's the whole workflow: point it at a text file, tell it where to save
the result. Useful flags:

| Flag | Default | What it controls |
|---|---|---|
| `-d, --data-set <path>` | `training-data.txt` | Text file to train on |
| `-e, --epochs <n>` | 10 | Full passes over the data |
| `--lr <n>` | 0.1 | Peak learning rate (cosine-decays to 10% of this by the last epoch) |
| `--vocab-size <n>` | 8000 | Max distinct tokens; rest collapse to `<unk>` |
| `--embed-dim <n>` | 16 | Size of each token's learned embedding |
| `--hidden-dim <n>` | 96 | Size of the GRU's hidden state (memory capacity) |
| `--seq-len <n>` | 25 | Truncated-BPTT window: how far back gradients flow |
| `-b, --batch-size <n>` | 512 | Sequences per parallel mini-batch |
| `--max-chars <n>` | — | Truncate input to the first N characters |
| `--load-model <path>` | — | Resume from a checkpoint instead of random init (see [How you use a model](#4-how-you-use-a-model)) |
| `--save-model <path>` | — | Where to write the trained checkpoint |
| `--checkpoint-every <n>` | — | Also save every N epochs during training, not just at the end — real crash-safety for long runs |
| `-l, --length <n>` | 80 | Tokens to generate per sample, after training |
| `--prompt <text>` | — | Seed generation with this text instead of a bare newline |
| `--seed <n>` | — | Seed the sampling RNG for reproducible output (omit for fresh randomness each run) |
| `--temperature <n>` | 1.0 | <1 sharpens toward likely tokens, >1 flattens toward uniform |
| `--top-k <n>` | — | Restrict sampling to the K most probable tokens each step |
| `--top-p <n>` | — | Nucleus sampling: restrict to the smallest set covering probability mass `n` |
| `--analyze` | off | Don't train — measure and estimate instead (see below) |
| `--diagnose-saturation` | off | Don't train/generate — report how saturated the model's gates are on real data |

**Before committing to a long run**, use `--analyze`:

```sh
./target/release/tinylm --data-set corpora/shakespeare.txt --analyze
```

This runs a handful of *real* gradient-computation batches — your data,
your hyperparameters, this machine — through the exact same parallel code
path training uses, times them, and extrapolates: a suggested `--epochs`
(targeting a ~2M-token-exposure budget across all epochs, floored at
`vocab_size / 200` so a bigger output layer — more classes to calibrate —
gets proportionally more epochs instead of being starved, capped at 400 so
tiny corpora don't run forever), an estimated per-epoch and total training
time, and a ready-to-run command. It's a live measurement, not a canned
formula — it stays honest even if you change `--hidden-dim`,
`--batch-size`, or run it on different hardware.

That vocab-scaled floor exists because of a real bug we hit and fixed:
the original flat floor of 15 epochs undertrained every corpus with a
large vocabulary — confirmed directly by comparing two runs where the
*larger*-vocab corpus got *more* total token exposure yet converged to a
much smaller fraction of the distance off its random-guessing baseline.
Symptom in practice: generated text nearly missing the most common words
in the language (`the`, `a`, `and`) because the model hadn't trained long
enough to learn even basic frequency statistics. If output reads as
unusually word-salady even by this project's standards, `--analyze` a
model you're unsure about and compare against how many epochs it actually
got.

## 4. How you use a model

Skip training entirely and generate from a saved checkpoint:

```sh
./target/release/tinylm --load-model models/macbeth.model --epochs 0 --length 200
```

`--load-model` restores the vocabulary and all learned weights — the
model's `embed-dim`/`hidden-dim` come from the checkpoint itself, not
your CLI flags, so you don't need to remember what you trained it with.
`--epochs 0` skips straight to generation. Omit it (or pass a positive
number) to keep training the loaded model further instead — checkpoints
are resumable.

Pretrained checkpoints in [`models/`](models/), one per corpus in
[`corpora/`](corpora/). All trained at `vocab_size=8000`/`hidden_dim=96`
(or full corpus vocab where the corpus has fewer unique words than that),
epoch count chosen per-corpus via `--analyze`:

| Model | Trained on |
|---|---|
| `models/shakespeare.model` | Complete works of Shakespeare (boilerplate-stripped) |
| `models/macbeth.model` | Macbeth only |
| `models/alice.model` | Alice's Adventures in Wonderland |
| `models/sherlock.model` | The Adventures of Sherlock Holmes |
| `models/grimm.model` | Grimm's Fairy Tales |
| `models/mothergoose.model` | The Real Mother Goose (nursery rhymes) |

## FAQ

**Can you prompt it, like a real LLM?**
Yes — `--prompt "some text"`. Mechanically, prompting means feeding tokens
through the model first, advancing its hidden state, *without* sampling
anything yet, then generating from wherever that leaves it. That's
exactly what `--prompt` does: it runs the prompt through the same
`step()` function every training step already uses, before the sampling
loop starts. Omit it and generation seeds from a single invisible
newline instead, same as before `--prompt` existed.

**Is output ever deterministic, or is "not the same twice" inherent to
how these models work?**
Determinism is a *choice*, not a property the model lacks. The forward
pass (`step()`) has zero randomness — same weights, same input token,
same hidden state → same output distribution, always. The *only*
randomness in the whole pipeline is the final sampling draw: which token
to emit from that distribution. `--seed <n>` fixes that draw, and it's
verified to work — running the same `--seed` twice produces byte-
identical output. (If you always took the highest-probability token
instead of sampling — greedy decoding — output would be 100%
deterministic with no seed needed at all.) Real LLM APIs work the same
way under the hood; "AI is unpredictable" is really describing the
sampling policy (temperature), not some inherent unpredictability in the
network itself.

**Does this actually qualify as an "LLM"? Does "large" mean anything
quantifiable?**
No official industry-wide cutoff exists, but there's rough consensus,
and we can just count. This model, at its default hyperparameters (8000
vocab, 16-dim embeddings, 96-dim hidden state), has **~937,000
parameters**. The smallest models people commonly call "large" (GPT-2
small, BERT-base) start around 110–125 *million* — roughly **130x more**
than this whole project. GPT-3 is ~187,000x bigger. Training data tells
the same story: this trains on ~1.2M tokens per corpus; modern LLMs
train on trillions. So no, this doesn't meet any reasonable quantitative
bar for "large," and it isn't supposed to — the name is the joke.

**Is it a real language model, or "just" an RNN?**
Both, not either/or. "RNN" names the *mechanism* (a recurrent hidden
state); "language model" names the *task* (assign a probability
distribution to "what token comes next, given everything before it").
Those are orthogonal — n-gram frequency tables, RNNs, LSTMs, and
Transformers are all "language models" in the technical sense; it's
architecture-agnostic terminology. RNN-based ones were literally called
that in the literature that helped start the whole neural-LM lineage
`model.rs`'s header comment traces (Mikolov et al., 2010, *"Recurrent
neural network based language model"*). So: a genuine, legitimate
(recurrent) neural language model — just not a large one.
