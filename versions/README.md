# TinyLM version history

This project went through several real architectural rewrites, each one forced by a concrete
limitation of the version before it — not refactors for their own sake. Each folder here is a
**runnable snapshot** in the same Rust codebase, deliberately, so you can `diff` between stages
instead of reading about the differences secondhand: v2 and v3 are frozen byte-for-byte from the
commits where those stages actually lived (real source, real pretrained checkpoint, untouched); v1
is a from-v2 reconstruction, trained fresh, because the project's real first version was TypeScript
(`tiny_llm.ts`, still in git history at commit [`414de75`](https://github.com/theraccoonbear/TinyLM/commit/414de75)) and a language switch on top of an architecture switch would have
buried the actual lesson each stage teaches. Every README's generated output is real and unedited —
nothing here is retrained or touched up after the fact.

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
