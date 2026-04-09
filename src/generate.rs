/// Text generation from a trained transformer model.
///
/// Given a prompt, we:
/// 1. Tokenize the prompt
/// 2. Feed it through the model to get next-token probabilities
/// 3. Sample from those probabilities (with temperature + top-k)
/// 4. Append the new token, repeat
use rand::Rng;

use crate::tensor::softmax_2d;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;

/// Generation parameters
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        GenerateConfig {
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
        }
    }
}

/// Generate text from a prompt
pub fn generate(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerateConfig,
) -> String {
    let mut rng = rand::thread_rng();

    // Tokenize the prompt
    let mut tokens = vec![tokenizer.bos_id()];
    tokens.extend(tokenizer.encode(prompt));

    let eos_id = tokenizer.eos_id();

    for _ in 0..config.max_tokens {
        // Truncate to max sequence length
        let start = if tokens.len() > model.config.max_seq_len {
            tokens.len() - model.config.max_seq_len
        } else {
            0
        };
        let input = &tokens[start..];

        // Forward pass
        let logits = model.forward(input);

        // Get logits for the last position
        let last_idx = logits.shape()[0] - 1;
        let last_logits = logits.slice(ndarray::s![last_idx..last_idx + 1, ..]).to_owned();

        // Apply temperature
        let scaled = &last_logits / config.temperature;
        let probs = softmax_2d(&scaled);
        let probs_row: Vec<f32> = probs.row(0).to_vec();

        // Top-k sampling
        let next_token = top_k_sample(&probs_row, config.top_k, &mut rng);

        if next_token == eos_id {
            break;
        }

        tokens.push(next_token);
    }

    // Decode, skipping the BOS token
    tokenizer.decode(&tokens[1..])
}

/// Sample from the top-k most probable tokens
fn top_k_sample(probs: &[f32], k: usize, rng: &mut impl Rng) -> u32 {
    // Get indices sorted by probability (descending)
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Keep only top k
    let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    // Renormalize
    let sum: f32 = top_k.iter().map(|(_, p)| p).sum();
    let normalized: Vec<(usize, f32)> = top_k.into_iter().map(|(i, p)| (i, p / sum)).collect();

    // Sample
    let r: f32 = rng.r#gen();
    let mut cumsum = 0.0;
    for (idx, prob) in &normalized {
        cumsum += prob;
        if r <= cumsum {
            return *idx as u32;
        }
    }

    // Fallback: return the most probable token
    normalized[0].0 as u32
}
