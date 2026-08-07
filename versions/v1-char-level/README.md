# v1 — character-level, no context

Rust, reconstructed from v2 by scaling it back: same MLP architecture, `context=1`, and a
character-level tokenizer in place of v2's word tokenizer. Trained fresh for this snapshot
(2026-08-06) — not extracted from history like v2/v3, since it never existed as its own Rust
commit. **The project's actual first prototype was TypeScript**, not this — see `tiny_llm.ts` at
commit [`414de75`](https://github.com/theraccoonbear/TinyLM/commit/414de75) in git history for
that literal original artifact. This file exists so the whole v1→v3 arc reads as one continuous
Rust codebase you can actually `diff` between stages, instead of a language switch stacked on top
of an architecture switch.

## What it does, and the actual point of this version

Structurally, this **is** v2's code — same `Model`, same `forward`/`accumulate_gradients`, same
embed→concat→W1,tanh→W2,softmax pipeline. Exactly two things changed:

1. `tokenize()` splits into individual **characters** instead of words (this also *simplifies* the
   tokenizer — no word/punctuation-attachment heuristics needed; `detokenize` is now an exact,
   lossless round-trip, just `tokens.concat()`).
2. `--context` defaults to **1** instead of 4 — the minimum meaningful window, so each prediction
   depends on only the single character immediately before it.

That's it. Nothing in the model itself needed to change — which is the actual lesson this version
demonstrates: a fixed-context architecture doesn't know or care what a "token" is or how wide the
window is. Order-1 character prediction and order-4 word prediction are the *same code path*. What
that architecture can't do, regardless of which of those two knobs you turn, is see further back
than `context` tokens — that ceiling is architectural, not a training problem, and it's exactly why
v3 (recurrent, unbounded context) exists.

## Real training run

Trained on the same corpus as v2's `shakespeare.model` — `corpora/shakespeare.txt` at the repo
root, confirmed byte-identical to the historical `shakespeare.clean.txt` used back at `414de75`.

```
cargo build --release
./target/release/tiny_llm --data-set ../../corpora/shakespeare.txt --epochs 50 --save-model shakespeare.model
```

- `vocab_size` (actual, not the cap): **84** — real English prose only ever uses on the order of
  70-100 distinct characters, so the `--vocab-size 3000` cap never binds here at all, unlike v2
  where it does real work.
- `context=1`, `embed_dim=16`, `hidden_dim=48`, `lr=0.1`, `batch_size=8192` (all v2's defaults,
  unchanged).
- 50 epochs, avg loss **3.26 → 2.47**, **416.05s** total (~8.3s/epoch — much cheaper per-epoch
  than v2's word-level training, since the output layer here is 84-wide instead of 3000-wide).

## Real output (this checkpoint, unedited, `--epochs 0 --length 150`)

```
 sho  drivis       the. o  sthathert wergiry, f  y patherayed d aldome  athalecaut as n s aiss  itss  aver
 ay t min tinin
  the   MOWe, P.    sevige 
---
 anedit  m  a  s  teles.
  ue ost thathithe m odot JLO.
 nowithe; ldis    O. Wh lles,  nche t tha  meliltitsthers: hay stins  beere ngh g cororin?
```

Compare this directly against [v2's real output](../v2-fixed-context-mlp/README.md) from the same
corpus: fragments like `the`, `thathert`, `beere` are recognizable letter-clusters and near-misses,
but essentially no real dictionary words survive, and there's no "Queen"/"CLOWN"/"Majesty" the way
v2 produces. Loss (2.47) is numerically *lower* than v2's (around 5) here, but that's an artifact
of vocabulary size, not model quality — cross-entropy over 84 classes is trivially lower than over
3000 classes, and "getting the vocabulary right" doesn't mean "producing real words" when the
vocabulary is single characters. This is the concrete version of the point the root README's
"Evolution" section makes in prose: word-level tokenization is what made real words possible at
all, not more training.

## What it couldn't do, and why v2 exists

Precisely nothing here can capture "word-ness" — the model's smallest unit of meaning is a single
character, so there's no representation for `hello` being a thing, no matter how much you train or
how wide you push `--context`. **v2** fixes this at the tokenizer level: tokens become words, and
the same architecture that's already sitting in this file suddenly has a chance to produce them.
