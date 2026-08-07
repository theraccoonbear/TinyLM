// A genuinely tiny language model: predicts the NEXT character from the
// CURRENT character, trained on real text via real backprop (not the
// finite-difference approximation from before — this is the actual math
// real LLMs use, just at a scale you can read in one file).

// ---------- 1. Tokenization ----------
// Real LLMs split text into subword "tokens" (chunks like "ing", "the").
// We're using single characters as tokens — same concept, simpler unit.

// Training data comes from an external file, so you can point this at
// anything (a book, source code, chat logs) without touching the code.
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";

// ---------- 0. Command-line arguments ----------
const USAGE = `
Usage: tsx tiny_llm.ts [options]

Options:
  -d, --data-set <path>   Training text file to learn from
                          (default: ./training-data.txt next to the script)
  -e, --epochs <n>        Training epochs (default: 300)
      --lr <n>            Learning rate (default: 0.1)
  -l, --length <n>        Characters to generate per sample (default: 40)
      --max-chars <n>     Truncate the data set to the first n characters
                          (handy for large files while you're experimenting)
  -h, --help              Show this help and exit

Examples:
  tsx tiny_llm.ts --data-set ./shakespeare.txt --max-chars 200000 --epochs 20
  tsx tiny_llm.ts -d ./my-notes.txt -e 500
`;

const cliOptions = {
    "data-set": { type: "string", short: "d" },
    "epochs": { type: "string", short: "e" },
    "lr": { type: "string" },
    "length": { type: "string", short: "l" },
    "max-chars": { type: "string" },
    "help": { type: "boolean", short: "h" },
} as const;

function parseCliArgs() {
    try {
        return parseArgs({
            args: process.argv.slice(2),
            allowPositionals: true, // `tsx tiny_llm.ts foo.txt` still works
            options: cliOptions,
        });
    } catch (err) {
        console.error((err as Error).message);
        console.error(USAGE);
        return process.exit(1); // never returns; satisfies the function's return type
    }
}
const args = parseCliArgs();

if (args.values.help) {
    console.log(USAGE);
    process.exit(0);
}

function parseNumberArg(name: string, raw: string | undefined, fallback: number): number {
    if (raw === undefined) return fallback;
    const n = Number(raw);
    if (Number.isNaN(n)) {
        console.error(`Invalid value for --${name}: "${raw}" is not a number`);
        console.error(USAGE);
        process.exit(1);
    }
    return n;
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const defaultDataPath = join(__dirname, "training-data.txt");
const dataSetArg = args.values["data-set"] ?? args.positionals[0];
const dataPath = dataSetArg ? resolve(dataSetArg) : defaultDataPath;

const epochs = parseNumberArg("epochs", args.values.epochs, 300);
const learningRate = parseNumberArg("lr", args.values.lr, 0.1);
const genLength = parseNumberArg("length", args.values.length, 40);
const maxChars = args.values["max-chars"] !== undefined
    ? parseNumberArg("max-chars", args.values["max-chars"], Infinity)
    : Infinity;

const fallbackText = "hello world hello there hello friend";
let trainingText: string;
if (existsSync(dataPath)) {
    trainingText = readFileSync(dataPath, "utf-8");
    console.log(`Loaded training data from: ${dataPath} (${trainingText.length} chars)`);
    if (trainingText.length > maxChars) {
        trainingText = trainingText.slice(0, maxChars);
        console.log(`Truncated to --max-chars ${maxChars} (${trainingText.length} chars)`);
    }
} else {
    trainingText = fallbackText;
    console.log(`No training file found at ${dataPath} — using built-in fallback text.`);
}

const vocabulary: string[] = [...new Set(trainingText)].sort();
const vocabSize = vocabulary.length;

function tokenToId(ch: string): number {
    return vocabulary.indexOf(ch);
}
function idToToken(id: number): string {
    return vocabulary[id];
}

console.log("Vocabulary (all unique characters seen):", vocabulary);
console.log("Vocab size:", vocabSize);

// ---------- 2. Model parameters ----------
// EMBEDDING: one trainable vector per token. This is the model's learned
// "meaning" for each character — starts as random noise, training shapes it.
const embedDim = 8;
let embedding: number[][] = vocabulary.map(() =>
    Array.from({ length: embedDim }, () => (Math.random() - 0.5) * 0.1)
);

// OUTPUT LAYER: turns an embedding vector into a raw score ("logit") per
// possible next character. Same dot-product-plus-bias idea as runNeuron,
// just vocabSize neurons running in parallel instead of one.
let W: number[][] = Array.from({ length: embedDim }, () =>
    Array.from({ length: vocabSize }, () => (Math.random() - 0.5) * 0.1)
);
let b: number[] = new Array(vocabSize).fill(0);

// ---------- 3. Forward pass ----------
function forward(tokenId: number): { logits: number[]; probs: number[] } {
    const e = embedding[tokenId]; // this token's embedding vector

    // logits[j] = e . W[:,j] + b[j]  — one score per possible next char
    const logits = new Array(vocabSize).fill(0);
    for (let j = 0; j < vocabSize; j++) {
        let sum = b[j];
        for (let k = 0; k < embedDim; k++) {
            sum += e[k] * W[k][j];
        }
        logits[j] = sum;
    }

    // softmax: turn raw scores into a probability distribution (sums to 1)
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit)); // -max for numeric stability
    const sumExps = exps.reduce((a, x) => a + x, 0);
    const probs = exps.map(x => x / sumExps);

    return { logits, probs };
}

// ---------- 4. Loss + backward pass (real gradients, not approximated) ----------
// Cross-entropy loss + softmax has a famously clean gradient:
// dLoss/dLogit[j] = probs[j] - (1 if j is the correct answer else 0)
function trainStep(inputId: number, targetId: number, learningRate: number): number {
    const { probs } = forward(inputId);
    const e = embedding[inputId];

    const dLogits = probs.slice();
    dLogits[targetId] -= 1; // the clean gradient formula above

    const loss = -Math.log(probs[targetId] + 1e-9);

    // Gradient for W and b: standard "outer product" of input and dLogits
    const dW: number[][] = Array.from({ length: embedDim }, () => new Array(vocabSize).fill(0));
    for (let k = 0; k < embedDim; k++) {
        for (let j = 0; j < vocabSize; j++) {
            dW[k][j] = e[k] * dLogits[j];
        }
    }
    const db = dLogits;

    // Gradient flowing back INTO the embedding (this is "backpropagation" —
    // the error signal passed backward through the layer it came from)
    const dE = new Array(embedDim).fill(0);
    for (let k = 0; k < embedDim; k++) {
        let sum = 0;
        for (let j = 0; j < vocabSize; j++) {
            sum += W[k][j] * dLogits[j];
        }
        dE[k] = sum;
    }

    // Apply updates — same "move opposite the gradient" rule as before
    for (let k = 0; k < embedDim; k++) {
        for (let j = 0; j < vocabSize; j++) {
            W[k][j] -= learningRate * dW[k][j];
        }
    }
    for (let j = 0; j < vocabSize; j++) {
        b[j] -= learningRate * db[j];
    }
    for (let k = 0; k < embedDim; k++) {
        embedding[inputId][k] -= learningRate * dE[k];
    }

    return loss;
}

// ---------- 5. Training loop ----------
// Build (current char -> next char) pairs from the real text
const pairs: [number, number][] = [];
for (let i = 0; i < trainingText.length - 1; i++) {
    pairs.push([tokenToId(trainingText[i]), tokenToId(trainingText[i + 1])]);
}

// This is a single-threaded, character-at-a-time toy — cost scales with
// pairs * epochs * embedDim * vocabSize. Big files + many epochs get slow
// fast; nudge people toward --max-chars / --epochs instead of a hang.
const estimatedOps = pairs.length * epochs * embedDim * vocabSize;
if (estimatedOps > 2_000_000_000) {
    console.log(
        `\n⚠ ${pairs.length} pairs × ${epochs} epochs is a lot for this toy trainer ` +
        `(~${(estimatedOps / 1e9).toFixed(1)}B ops) — it may take a very long time.\n` +
        `  Try: --max-chars 50000 to shrink the data set, and/or a smaller --epochs.\n`
    );
}

for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    for (const [inputId, targetId] of pairs) {
        totalLoss += trainStep(inputId, targetId, learningRate);
    }
    if (epoch % 50 === 0) {
        console.log(`epoch ${epoch}: avg loss = ${(totalLoss / pairs.length).toFixed(4)}`);
    }
}

// ---------- 6. Generation (autoregressive: feed output back in as next input) ----------
function generate(startChar: string, length: number): string {
    let result = startChar;
    let currentId = tokenToId(startChar);

    for (let i = 0; i < length; i++) {
        const { probs } = forward(currentId);

        // sample from the probability distribution rather than always
        // picking the top choice — this is why LLM output isn't identical
        // every time even for the same prompt
        let r = Math.random();
        let chosenId = 0;
        for (let j = 0; j < vocabSize; j++) {
            r -= probs[j];
            if (r <= 0) { chosenId = j; break; }
        }

        result += idToToken(chosenId);
        currentId = chosenId; // <-- the model's own output becomes its next input
    }
    return result;
}

console.log(`\n--- Generated text (trained on ${dataPath}) ---`);
const startChar = vocabulary.includes("h") ? "h" : vocabulary[0];
console.log(generate(startChar, genLength));
console.log(generate(startChar, genLength));
console.log(generate(vocabulary[Math.floor(vocabulary.length / 2)], genLength));
