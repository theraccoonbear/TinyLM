# TinyLM version history

This project went through several real architectural rewrites, each one forced by a concrete
limitation of the version before it — not refactors for their own sake. Each folder here is a
**frozen, runnable snapshot**: real source code and a real pretrained checkpoint extracted
byte-for-byte from the commit where that stage lived, plus a README with the exact params used and
real (unedited) generated output from that checkpoint. Nothing in these folders is retrained or
touched up — you're seeing exactly what that stage of the project actually produced.

Read them in order if you want to feel the arc rather than just read about it:

| | Stage | Architecture | What it added | What it still couldn't do |
|---|---|---|---|---|
| [v1](v1-char-level/) | char-level | next-char prediction, no context | the starting point | no concept of a word |
| [v2](v2-fixed-context-mlp/) | fixed-context MLP | word embeddings + 4-token window | real words, real vocabulary | context hard-capped at 4 tokens — an architectural ceiling |
| [v3](v3-gru-sgd/) | recurrent GRU, plain SGD | a real hidden state + backprop-through-time | unbounded context, no more window ceiling | SGD converged too slowly/unreliably, and nothing caught it when it did |
| v4 | *(repo root — you're already there)* | recurrent GRU, Adam | reliable convergence + an automated sanity check/retry loop that catches undertraining instead of shipping it | see the root [README](../README.md) for what's still open |

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
