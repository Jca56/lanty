/// Byte-Pair Encoding tokenizer, built from scratch.
///
/// How BPE works:
/// 1. Start with individual bytes as tokens
/// 2. Count all adjacent pairs of tokens in the training data
/// 3. Merge the most frequent pair into a new token
/// 4. Repeat until we reach the desired vocabulary size
///
/// This lets the model work with subword units - common words become single tokens,
/// rare words get split into pieces.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct Tokenizer {
    /// Maps token string -> token ID
    pub vocab: HashMap<String, u32>,
    /// Maps token ID -> token string
    pub id_to_token: HashMap<u32, String>,
    /// Ordered list of merge rules (pair -> merged token)
    pub merges: Vec<(String, String)>,
    /// Total vocabulary size
    pub vocab_size: u32,
}

/// Special tokens
pub const PAD_TOKEN: &str = "<PAD>";
pub const UNK_TOKEN: &str = "<UNK>";
pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

impl Tokenizer {
    /// Train a BPE tokenizer from text
    pub fn train(text: &str, vocab_size: u32, progress: bool) -> Self {
        let bar = if progress {
            let bar = indicatif::ProgressBar::new(vocab_size as u64);
            bar.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} merges")
                    .unwrap()
                    .progress_chars("##-"),
            );
            Some(bar)
        } else {
            None
        };

        // Start with byte-level tokens (0-255) plus special tokens
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: HashMap<u32, String> = HashMap::new();
        let mut next_id: u32 = 0;

        // Add special tokens first
        for special in &[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] {
            vocab.insert(special.to_string(), next_id);
            id_to_token.insert(next_id, special.to_string());
            next_id += 1;
        }

        // Add all single bytes
        for byte in 0..=255u8 {
            let s = format!("{:02x}", byte);
            vocab.insert(s.clone(), next_id);
            id_to_token.insert(next_id, s);
            next_id += 1;
        }

        // Convert text to byte sequences (each word becomes a list of hex byte tokens)
        // We split on whitespace boundaries, keeping the space attached to the next word
        let words = split_into_words(text);
        let mut word_tokens: Vec<(Vec<String>, usize)> = Vec::new();
        let mut word_counts: HashMap<Vec<String>, usize> = HashMap::new();

        for word in &words {
            let bytes: Vec<String> = word.bytes().map(|b| format!("{:02x}", b)).collect();
            *word_counts.entry(bytes).or_insert(0) += 1;
        }

        for (tokens, count) in &word_counts {
            word_tokens.push((tokens.clone(), *count));
        }

        let mut merges: Vec<(String, String)> = Vec::new();

        if let Some(ref bar) = bar {
            bar.set_position(next_id as u64);
        }

        // Perform merges until we reach desired vocab size
        while next_id < vocab_size {
            // Count all adjacent pairs
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (tokens, count) in &word_tokens {
                if tokens.len() < 2 {
                    continue;
                }
                for i in 0..tokens.len() - 1 {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += count;
                }
            }

            // Find most frequent pair
            let best = pair_counts.into_iter().max_by_key(|&(_, count)| count);
            let (best_pair, _count) = match best {
                Some(p) => p,
                None => break, // No more pairs to merge
            };

            // Create new merged token
            let merged = format!("{}{}", best_pair.0, best_pair.1);
            vocab.insert(merged.clone(), next_id);
            id_to_token.insert(next_id, merged.clone());
            merges.push(best_pair.clone());
            next_id += 1;

            // Apply this merge to all word tokens
            let mut new_word_tokens = Vec::new();
            for (tokens, count) in &word_tokens {
                let new_tokens = apply_merge(tokens, &best_pair.0, &best_pair.1, &merged);
                new_word_tokens.push((new_tokens, *count));
            }
            word_tokens = new_word_tokens;

            if let Some(ref bar) = bar {
                bar.set_position(next_id as u64);
            }
        }

        if let Some(bar) = bar {
            bar.finish_with_message("Tokenizer training complete!");
        }

        Tokenizer {
            vocab,
            id_to_token,
            merges,
            vocab_size: next_id,
        }
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let words = split_into_words(text);
        let mut all_ids = Vec::new();

        for word in &words {
            let mut tokens: Vec<String> = word.bytes().map(|b| format!("{:02x}", b)).collect();

            // Apply merges in order
            for (a, b) in &self.merges {
                let merged = format!("{}{}", a, b);
                tokens = apply_merge(&tokens, a, b, &merged);
            }

            // Convert to IDs
            let unk_id = self.vocab[UNK_TOKEN];
            for token in &tokens {
                let id = self.vocab.get(token).copied().unwrap_or(unk_id);
                all_ids.push(id);
            }
        }

        all_ids
    }

    /// Decode token IDs back into text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut hex_string = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Skip special tokens in output
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                hex_string.push_str(token);
            }
        }

        // Convert hex pairs back to bytes
        let mut bytes = Vec::new();
        let chars: Vec<char> = hex_string.chars().collect();
        let mut i = 0;
        while i + 1 < chars.len() {
            let hex: String = chars[i..i + 2].iter().collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                bytes.push(byte);
                i += 2;
            } else {
                i += 1;
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Save tokenizer to a JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load tokenizer from a JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let tok: Tokenizer = serde_json::from_str(&json)?;
        Ok(tok)
    }

    /// Get the BOS token ID
    pub fn bos_id(&self) -> u32 {
        self.vocab[BOS_TOKEN]
    }

    /// Get the EOS token ID
    pub fn eos_id(&self) -> u32 {
        self.vocab[EOS_TOKEN]
    }
}

/// Split text into words, keeping leading whitespace attached
fn split_into_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_word = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if in_word {
                words.push(current.clone());
                current.clear();
            }
            current.push(ch);
            in_word = false;
        } else {
            if !in_word && !current.is_empty() {
                // Attach whitespace to the start of this word
                current.push(ch);
                in_word = true;
            } else {
                current.push(ch);
                in_word = true;
            }
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Apply a single merge operation to a token list
fn apply_merge(tokens: &[String], a: &str, b: &str, merged: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
            result.push(merged.to_string());
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let text = "hello world hello world hello world foo bar baz";
        let tok = Tokenizer::train(text, 280, false);
        let encoded = tok.encode("hello world");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_special_tokens() {
        let tok = Tokenizer::train("test", 270, false);
        assert!(tok.vocab.contains_key(PAD_TOKEN));
        assert!(tok.vocab.contains_key(UNK_TOKEN));
        assert!(tok.vocab.contains_key(BOS_TOKEN));
        assert!(tok.vocab.contains_key(EOS_TOKEN));
    }
}
