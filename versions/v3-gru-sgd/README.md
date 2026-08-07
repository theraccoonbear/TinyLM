# v3 — recurrent GRU, plain SGD

Frozen from commit [`d4d7203`](https://github.com/theraccoonbear/TinyLM/commit/d4d7203) (2026-08-05 22:54:50 -0500) — the first GRU commit, before the module split and before the
switch to Adam.

## What it does

Replaces v2's fixed 4-token window with a real **GRU (Gated Recurrent Unit)** hidden state,
updated one token at a time and trained via **truncated backprop-through-time** (`--seq-len 25`):
the update gate, reset gate, and candidate state are hand-derived and hand-differentiated — no
autodiff library — verified by a numerical gradient-check test (`gru_gradients_match_finite_
differences`) that's still in the codebase today, unchanged since this commit.

```
z = sigmoid(Wz·x + Uz·h + bz)              (update gate)
r = sigmoid(Wr·x + Ur·h + br)              (reset gate)
h~ = tanh(Wh·x + Uh·(r⊙h) + bh)            (candidate state)
h' = (1-z)⊙h + z⊙h~                        (new hidden state)
```

Defaults at this commit: `--epochs 10`, `--lr 0.1`, `--vocab-size 3000`, `--embed-dim 16`,
`--hidden-dim 48`, `--seq-len 25`, `--batch-size 512`. **This `--lr 0.1` matters** — it's a
plain-SGD learning rate, an order of magnitude+ larger than the `0.003` the codebase uses today,
because SGD (unlike Adam) has no per-parameter adaptive step size of its own.

There is no automated sanity check yet, no cleaned corpora yet (front-matter/ToC still present),
and no retry-on-undertraining loop — all three were added later, specifically *because* of
problems this version's plain-SGD training turned out to have (see below).

Five checkpoints from this commit are included as-committed: `alice`, `grimm`, `macbeth`,
`mothergoose`, `sherlock`. A sixth, `shakespeare`, is included too but wasn't part of this commit
— see below for exactly where it actually comes from.

## The shakespeare checkpoint, and why it's not from this commit

This commit never trained on Shakespeare at all — `d4d7203`'s diff shows the *previous*
architecture's `shakespeare.model` (v2's, 833,014 bytes) being deleted, and nothing replacing it.
The first GRU-era Shakespeare run happened 29 minutes later, in commit
[`1a6fa9b`](https://github.com/theraccoonbear/TinyLM/commit/1a6fa9b) — still plain SGD, still
`vocab_size=3000`/`hidden_dim=48` (confirmed by checkpoint byte size: 858,022 bytes, right in the
same range as this folder's other five checkpoints; the *next* morning's commit bumped those
defaults to 8000/96, which is what today's root `models/shakespeare.model` uses instead). The only
code that changed between `d4d7203` and `1a6fa9b` is CLI-surface additions (`--analyze`,
`--prompt`, `--seed`) — the `Model` struct and its forward/backward math are untouched, so this
checkpoint loads and runs correctly against this exact frozen binary. Extracted byte-for-byte via
`git show 1a6fa9b:models/shakespeare.model`, verified against the blob hash before being added
here.

**15 epochs, avg loss 7.74 → 5.44, 874.5s** — plain SGD, on the same `corpora/shakespeare.txt`
used by [v1](../v1-char-level/) and [v2](../v2-fixed-context-mlp/), which is what makes this the
real cross-version comparison point rather than `mothergoose`/`alice`/etc. below (those never had
a v1/v2 equivalent trained on the same text). See the [reference machine](../README.md#reference-machine)
for what "874.5s" is relative to — this specific number is from the original 2026-08-05 run, not
re-measured on that machine, since (unlike v2's missing timing) it was already recorded honestly
at the time.

```
./target/release/tiny_llm --load-model models/shakespeare.model --epochs 0 --length 150
```

```
Loaded model from models/shakespeare.model (vocab 3000, embed_dim 16, hidden_dim 48)
--epochs 0: skipping training, using the model as loaded.

--- Generated text ---
courtesy whither died <unk> PROTEUS And <unk> proper time <unk>
.-
Highness I I silver. Which, so Dog. <unk>. <unk> deserv'd in I <unk> in quarrel,. Orleans Don me few <unk> you <unk>?
```

Real character names spanning the whole canon (PROTEUS, Capulet, JUSTICE, AENEAS, BIONDELLO,
Roderigo, KATHERINA, CRESSIDA, Posthumus, LEONTES, HOLOFERNES...) — recurrence is clearly doing
*something* — but still mostly word-salad at 15 epochs. Directly comparable to
[v1's](../v1-char-level/) 50-epoch/416s and [v2's](../v2-fixed-context-mlp/) 15-epoch/822s runs on
the identical text: this is the real cost of *this* architecture reaching *this* point, on *this*
corpus, no cherry-picking.

## Run it yourself

```
cd versions/v3-gru-sgd
cargo build --release
./target/release/tiny_llm --load-model models/mothergoose.model --epochs 0 --length 100
```

## Real output (this run, this checkpoint, unedited)

```
Loaded model from models/mothergoose.model (vocab 3000, embed_dim 16, hidden_dim 48)
--epochs 0: skipping training, using the model as loaded.

--- Generated text ---
wood lasses bid plates
potato
paper, BLUE farmer least corn like MARY'S GROUND and fourteen Alphabet dearly nine, Built the chamber crept DEATH ship playfellows, brother's SAVING Baby
peep leaping Ann meet whistle Thread, Build
```

Real words, real recurrence (no `<unk>`-per-4-tokens ceiling like v2), but still not coherent —
this is a genuinely undertrained model, which is the actual point of keeping it around: recurrence
alone didn't fix quality, *training* did, and figuring out why took real diagnosis.

## What it couldn't do, and why the current codebase exists

Recurrence solved the architectural ceiling from v2, but plain SGD converged too slowly and
unreliably to depend on, and there was nothing in the pipeline to catch it when it did. The
concrete proof, from later in the project's history (commit `7be21d0`): training `macbeth` with
plain SGD for **78 epochs** landed at loss **7.81** and *failed* an automated word-frequency
sanity check; training the same corpus with **Adam** for just **30 epochs** landed at loss
**6.05** and passed. Same architecture, same shape, same `model.rs` — the only thing that changed
was *how* the weights were searched for.

That, plus repeatedly finding undertrained models only by manually grepping generated text for
missing common words, is what motivated everything the current codebase (the repo root, one level
up from here) adds on top of this: the Adam optimizer, an automated post-training sanity check
with a distinct failing exit code, a retry loop that resumes training until that check actually
passes, and (separately) cleaning the front-matter/table-of-contents boilerplate out of several
corpora. None of that changed the model's shape — see the repo root's own README for what's
current.
