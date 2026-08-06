// ---------- Tokenizer ----------
// Turns raw text into tokens, and back again. Nothing in this file knows
// anything about neural networks — it's pure text processing, and the
// vocabulary built here is what gives token ids their meaning everywhere
// else in the crate.

use std::collections::HashMap;

// Splits into: runs of letters/apostrophes as one word token ("thou'lt"),
// each other punctuation character as its own token, and '\n' as its own
// token (so the model can learn *where lines end* — the closest thing to
// verse structure it can pick up). Plain whitespace is just a separator.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphabetic() || c == '\'' {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            tokens.push(std::mem::take(&mut word));
        }
        if c == '\n' {
            tokens.push("\n".to_string());
        } else if !c.is_whitespace() {
            tokens.push(c.to_string());
        }
    }
    if !word.is_empty() {
        tokens.push(word);
    }
    tokens
}

// Punctuation that hugs the token before it, with no space in between.
fn attaches_to_previous(tok: &str) -> bool {
    matches!(tok, "," | "." | "!" | "?" | ";" | ":" | "'" | ")" | "]" | "}" | "-")
}

// Reconstruct readable text from a token stream. Heuristic, not a real
// detokenizer, but good enough to make output look like actual prose/verse
// instead of a token-per-line dump.
pub fn detokenize(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut prev: Option<&str> = None;
    for tok in tokens {
        if tok == "\n" {
            out.push('\n');
        } else {
            let need_space = match prev {
                None => false,
                Some("\n") => false,
                Some(p) if matches!(p, "(" | "[" | "{" | "`") => false,
                _ if attaches_to_previous(tok) => false,
                _ => true,
            };
            if need_space {
                out.push(' ');
            }
            out.push_str(tok);
        }
        prev = Some(tok.as_str());
    }
    out
}

// Keep the `max_vocab - 1` most frequent tokens (reserving slot 0 for
// <unk>), then sort alphabetically for a readable listing. Everything not
// in this vocab maps to <unk> at lookup time — standard for bounding an
// output layer's cost against a Zipfian real-text vocabulary.
pub fn build_vocab(tokens: &[String], max_vocab: usize) -> (Vec<String>, HashMap<String, usize>) {
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        *freq.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut counted: Vec<(&str, usize)> = freq.into_iter().collect();
    counted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let keep = max_vocab.saturating_sub(1);
    let mut top: Vec<String> = counted.into_iter().take(keep).map(|(s, _)| s.to_string()).collect();
    top.sort();

    let mut vocab = Vec::with_capacity(top.len() + 1);
    vocab.push("<unk>".to_string());
    vocab.append(&mut top);

    let token_to_id = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    (vocab, token_to_id)
}
