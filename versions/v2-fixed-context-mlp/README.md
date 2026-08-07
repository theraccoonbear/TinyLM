# v2 — word-level, fixed-context MLP

Frozen from commit [`414de75`](https://github.com/theraccoonbear/TinyLM/commit/414de75) (2026-08-05 18:38:55 -0500) — the `rust/` half of the same initial commit as v1. The Rust
rewrite was word-level from its very first line; it's the *tokenization/context* model that
changed here, not the language.

## What it does

A Bengio-2003-style feedforward network: embed the last **4** tokens, concatenate those
embeddings into one vector, run it through a hidden layer (tanh) and a softmax output layer.

```
[tok(t-4) tok(t-3) tok(t-2) tok(t-1)] --embed--> concat --W1,tanh--> hidden --W2,softmax--> P(tok t)
```

Real words now exist as first-class units (via the embedding table + vocab), and the model can
condition on up to 4 tokens of history instead of 1 character — a genuine step up from v1. The
committed checkpoint (`shakespeare.model`, trained then, loaded as-is here) used `vocab_size=3000`,
`context=4`, `embed_dim=16`, `hidden_dim=48`, trained 15 epochs to average loss 7.38 → 5.13.

## Training cost

The original commit that produced this checkpoint never recorded wall-clock time. Rather than
guess, this same recipe (identical flags, same `corpora/shakespeare.txt`) was re-run fresh on the
[reference machine](../README.md#reference-machine) to measure it — that fresh run is *not* the
checkpoint shipped above (it's not committed anywhere; the shipped checkpoint stays untouched,
frozen from `414de75`), it's purely a timing measurement:

```
cd versions/v2-fixed-context-mlp
./target/release/tiny_llm --data-set ../../corpora/shakespeare.txt --epochs 15 --save-model /tmp/timing-only.model
```

15 epochs, avg loss **7.52 → 5.12** (random init means this isn't bit-identical to the shipped
checkpoint's 7.38 → 5.13, but it's the same recipe landing in the same place) — **821.56s** total.

## Run it yourself

```
cd versions/v2-fixed-context-mlp
cargo build --release
./target/release/tiny_llm --load-model shakespeare.model --epochs 0 --length 100
```

`--epochs 0` with `--load-model` skips training entirely and just samples from the checkpoint
exactly as it was originally trained — no retraining happened to produce the output below.

## Real output (this run, this checkpoint, unedited)

```
Loaded model from shakespeare.model (vocab 3000, context 4, embed_dim 16, hidden_dim 48)
--epochs 0: skipping training, using the model as loaded.

--- Generated text ---
Pray. part Queen up there
pleas'd <unk> <unk> my <unk>
The <unk>, And <unk> leave a <unk>;
ill, <unk> lords,
hell from they'll dote in'
<unk>. her King of of not <unk>.
CLOWN. Come she
I <unk>, upon doubt my love is For make thee?
```

Real words this time (`Pray`, `Queen`, `love`, `CLOWN`) — a genuine jump from v1 — but still
mostly noise: short local phrases string together, then collapse, because the model can never see
past the 4 tokens immediately behind it. There's no way for it to "remember" anything from earlier
in a sentence, let alone earlier in a line.

## What it couldn't do, and why v3 exists

The context window here is a hard architectural ceiling: `context=4` isn't a training
hyperparameter you can just crank up for better results — the input layer's width is *fixed* at
`4 * embed_dim`, baked into the model's shape. No amount of additional training escapes it.
**v3** replaces the fixed window with a **recurrent hidden state (GRU)** that carries information
forward indefinitely, trained via real backprop-through-time.
