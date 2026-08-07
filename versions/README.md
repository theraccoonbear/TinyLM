# TinyLM version history

This project went through several real architectural rewrites, each one forced by a concrete
limitation of the version before it — not refactors for their own sake. Each folder here is a
**runnable snapshot** in the same Rust codebase, deliberately, so you can `diff` between stages
instead of reading about the differences secondhand: v2 and v3 are frozen byte-for-byte from the
commits where those stages actually lived (real source, real pretrained checkpoints, untouched —
with one exception noted in v3's own README: its `shakespeare.model` is extracted from a commit 29
minutes later, since the freeze commit itself never trained on Shakespeare at all); v1 is a from-v2
reconstruction, trained fresh, because the project's real first version was TypeScript
(`tiny_llm.ts`, still in git history at commit [`414de75`](https://github.com/theraccoonbear/TinyLM/commit/414de75)) and a language switch on top of an architecture switch would have
buried the actual lesson each stage teaches. Every README's generated output is real and unedited —
nothing here is retrained or touched up after the fact.

Want to try one without installing Rust? Every stage below (plus the current
root version) is built and attached to the [Releases page][releases] for
Linux, macOS, and Windows — download, extract, run. Every pretrained checkpoint in this project
(all four stages, all corpora) is attached there too, individually, if you just want the weights
without cloning anything.

[releases]: https://github.com/theraccoonbear/TinyLM/releases

Read them in order if you want to feel the arc rather than just read about it:

| | Stage | Architecture | What it added | What it still couldn't do |
|---|---|---|---|---|
| [v1](v1-char-level/) | char-level | next-char prediction, no context | the starting point | no concept of a word |
| [v2](v2-fixed-context-mlp/) | fixed-context MLP | word embeddings + 4-token window | real words, real vocabulary | context hard-capped at 4 tokens — an architectural ceiling |
| [v3](v3-gru-sgd/) | recurrent GRU, plain SGD | a real hidden state + backprop-through-time | unbounded context, no more window ceiling | SGD converged too slowly/unreliably, and nothing caught it when it did |
| v4 | *(repo root — you're already there)* | recurrent GRU, Adam | reliable convergence + an automated sanity check/retry loop that catches undertraining instead of shipping it | see the root [README](../README.md) for what's still open |

v1 and v2 are, structurally, the *same code* — v1 is v2 with `context=1` and a character tokenizer
instead of a word one. Worth reading v1's README first for why that's true and what it demonstrates
on its own.

## What it actually costs to get here

Every version above trained a checkpoint on the exact same text — `corpora/shakespeare.txt`, the
complete works, unmodified between stages — which is what makes the following numbers a genuine
apples-to-apples comparison instead of four different anecdotes:

| | Params | Epochs | Loss | Wall-clock |
|---|---|---|---|---|
| [v1](v1-char-level/) | vocab=84 (uncapped), hidden=48, context=1 | 50 | 3.26 → 2.47 | **416.05s** |
| [v2](v2-fixed-context-mlp/) | vocab=3000, hidden=48, context=4 | 15 | 7.52 → 5.12 | **821.56s** |
| [v3](v3-gru-sgd/) | vocab=3000, hidden=48, GRU, plain SGD | 15 | 7.74 → 5.44 | **874.5s** |
| [v4](../README.md) | vocab=8000, hidden=96, GRU, Adam | 20 | 6.48 → 5.06 | **10863.21s** (~3.02hr) |

All four ran on the same [reference machine](#reference-machine). Read the table honestly, though:
it is **not** a single-variable comparison. v4's jump in wall-clock isn't purely "Adam instead of
SGD" — it's also running at more than double v3's hidden-layer width and more than double its
vocabulary cap, both of which cost real compute independent of the optimizer. The one genuinely
clean single-variable comparison in this project is in [v3's own README](v3-gru-sgd/#what-it-couldnt-do-and-why-the-current-codebase-exists):
`macbeth` trained at *matched* config, SGD vs. Adam, 78 epochs/loss 7.81/failed sanity vs. 30
epochs/loss 6.05/passed. That one isolates the optimizer. The table above isolates the *corpus* —
useful for a different question ("what did each stage of this project actually cost to run"), not
the same question.

v1's and v2's numbers are fresh runs done specifically to build this table (v2's original commit
never recorded a time at all). v3's and v4's are the real historical numbers from when those
checkpoints were actually trained, unchanged.

## Reference machine

All timings above (and the ones documented in the root README) are from the same box, so they're
at least internally comparable, though it's a shared desktop, not an isolated benchmarking
rig — treat these as "real numbers, same hardware," not laboratory-clean isolated measurements.

- **CPU**: Intel Core i7-6700 @ 3.40GHz — 4 cores / 8 threads, boost to 4.0GHz
- **RAM**: 15GB
- **OS**: Fedora Linux 42, running in a rootless Podman container — confirmed via `cgroup
  memory.max`/`cpu.max` (both report `max`, i.e. no limit) that the container imposes no CPU or
  memory ceiling below what the host actually has, so these are real numbers, not
  sandbox-throttled ones
- **Rust**: rustc 1.97.1, `--release` (`opt-level = 3, lto = true, codegen-units = 1`) — identical
  profile for every version above, no version got a faster build config than another

## The one-sentence version of each jump

- **v1 → v2**: characters can't capture "word-ness" no matter how much context you give them —
  switch to word-level tokenization with a learned embedding table.
- **v2 → v3**: a fixed context window is a ceiling baked into the model's *shape* — no amount of
  training escapes it. Switch to a recurrent hidden state (GRU) with real backprop-through-time,
  so context length becomes a training-time choice (`--seq-len`) instead of an architectural limit.
- **v3 → v4 (current)**: this jump is different in kind from the other two — the model's *shape*
  didn't change at all, only how it's trained. Plain SGD proved too slow/unreliable to trust
  (macbeth: 78 SGD epochs → loss 7.81 → failed sanity check, vs. 30 Adam epochs → loss 6.05 →
  passed), and there was no automated way to catch an undertrained model before it shipped. Adam,
  the sanity check, and the retry loop are what closed that gap — an optimizer and pipeline
  upgrade, not a new architecture.

## Why "the runner" looks unchanged from v3 onward

If you diff `model.rs`'s `Model` struct between v3 and the current root, the shape is identical —
same fields, same dimensions, same forward pass. That's expected: v3 → v4 was a training-strategy
change, not an architecture change, so the thing doing the training (`apply_gradients`) is the only
place the code actually differs. Compare that to v1 → v2 → v3, where the struct being trained is
a genuinely different shape every time (no embedding at all → embedding + fixed concat window →
embedding + recurrent `U` matrices that don't exist in v2's shape). "Same runner, different
values" is the right mental model for v3 vs. v4; it's the wrong one for v1 vs. v2 vs. v3.
