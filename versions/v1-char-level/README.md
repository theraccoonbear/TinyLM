# v1 — character-level, no context

Frozen from commit [`414de75`](https://github.com/theraccoonbear/TinyLM/commit/414de75) (2026-08-05 18:38:55 -0500), the very first commit — `tiny_llm.ts`.

## What it does

Predicts the **next character** from the **current character**. That's the entire model: no
embedding, no hidden state, no window of prior context wider than one character. It's closer to
a learned Markov transition table than anything you'd call a language model — included here as
the actual floor everything else in this project was built up from.

## Run it yourself

```
cd versions/v1-char-level
npm install
npx tsx tiny_llm.ts --data-set training-data.txt --epochs 300 --length 40
```

`training-data.txt` at this commit is a toy corpus — literally `hello world hello there hello
friend` (36 bytes) — this predates any of the real public-domain corpora used later.

## Real output (this run, this checkpoint, unedited)

```
he thererie ttheriend thelorend helorielo
held friereld hend frlo helo helld wo hen
ld he frlo frie wo f fre henld frerld the
```

That's what "next-character prediction with no context" gets you: locally plausible letter
transitions (real English digraphs/trigraphs — `th`, `he`, `nd`, `ld`) that never resolve into an
actual word, because the model is never looking further back than one character to decide what
comes next.

## What it couldn't do, and why v2 exists

There's no mechanism here for "word-ness" to emerge — the model has no concept of a token wider
than one character, so it can't learn that `hello` is a unit. **v2** fixes this by switching to
**word-level tokenization** with a real embedding table and a multi-token context window.
