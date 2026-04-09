/// The transformer model, built from scratch.
///
/// Architecture (same core idea as GPT):
///   Input tokens
///     -> Token embedding + Positional encoding
///     -> N transformer blocks, each containing:
///        -> Multi-head self-attention (with causal mask)
///        -> Feed-forward network (linear -> GELU -> linear)
///        -> Layer normalization + residual connections
///     -> Final layer norm
///     -> Linear projection to vocabulary logits
use ndarray::{Array1, Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::tensor::*;

/// Model hyperparameters
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub d_model: usize,      // embedding dimension
    pub n_heads: usize,       // number of attention heads
    pub n_layers: usize,      // number of transformer blocks
    pub d_ff: usize,          // feed-forward hidden dimension
    pub max_seq_len: usize,   // maximum sequence length
    pub dropout: f32,         // dropout rate (used during training)
}

impl ModelConfig {
    /// A small config suitable for training on a single GPU
    pub fn small(vocab_size: u32) -> Self {
        ModelConfig {
            vocab_size,
            d_model: 256,
            n_heads: 8,
            n_layers: 6,
            d_ff: 1024,
            max_seq_len: 256,
            dropout: 0.1,
        }
    }

    /// A tiny config for quick testing
    pub fn tiny(vocab_size: u32) -> Self {
        ModelConfig {
            vocab_size,
            d_model: 128,
            n_heads: 4,
            n_layers: 4,
            d_ff: 512,
            max_seq_len: 128,
            dropout: 0.1,
        }
    }

    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size as usize * self.d_model;
        let per_layer = {
            // Attention: Q, K, V, O projections
            let attn = 4 * self.d_model * self.d_model + 4 * self.d_model;
            // Feed-forward
            let ff = self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model;
            // Layer norms (2 per layer)
            let ln = 4 * self.d_model;
            attn + ff + ln
        };
        let final_ln = 2 * self.d_model;
        let output_proj = self.vocab_size as usize * self.d_model;
        embed + self.n_layers * per_layer + final_ln + output_proj
    }
}

/// A single attention head's parameters
#[derive(Serialize, Deserialize, Clone)]
pub struct AttentionParams {
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    pub b_q: Array1<f32>,
    pub b_k: Array1<f32>,
    pub b_v: Array1<f32>,
}

/// Multi-head attention parameters
#[derive(Serialize, Deserialize, Clone)]
pub struct MultiHeadAttentionParams {
    pub heads: Vec<AttentionParams>,
    pub w_o: Array2<f32>,
    pub b_o: Array1<f32>,
}

/// Feed-forward network parameters
#[derive(Serialize, Deserialize, Clone)]
pub struct FeedForwardParams {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

/// A single transformer block's parameters
#[derive(Serialize, Deserialize, Clone)]
pub struct TransformerBlockParams {
    pub attn: MultiHeadAttentionParams,
    pub ff: FeedForwardParams,
    pub ln1_gamma: Array1<f32>,
    pub ln1_beta: Array1<f32>,
    pub ln2_gamma: Array1<f32>,
    pub ln2_beta: Array1<f32>,
}

/// The full transformer model
#[derive(Serialize, Deserialize, Clone)]
pub struct Transformer {
    pub config: ModelConfig,
    pub token_embedding: Array2<f32>,
    pub blocks: Vec<TransformerBlockParams>,
    pub final_ln_gamma: Array1<f32>,
    pub final_ln_beta: Array1<f32>,
    pub output_projection: Array2<f32>,
}

impl Transformer {
    /// Initialize a new model with random weights
    pub fn new(config: &ModelConfig) -> Self {
        let d = config.d_model;
        let d_head = d / config.n_heads;
        let std_dev = (2.0f32 / d as f32).sqrt();
        let normal = Normal::new(0.0f32, std_dev).unwrap();

        let token_embedding = Array2::random((config.vocab_size as usize, d), normal);

        let mut blocks = Vec::new();
        for _ in 0..config.n_layers {
            let mut heads = Vec::new();
            for _ in 0..config.n_heads {
                heads.push(AttentionParams {
                    w_q: Array2::random((d, d_head), normal),
                    w_k: Array2::random((d, d_head), normal),
                    w_v: Array2::random((d, d_head), normal),
                    b_q: Array1::zeros(d_head),
                    b_k: Array1::zeros(d_head),
                    b_v: Array1::zeros(d_head),
                });
            }

            let attn = MultiHeadAttentionParams {
                heads,
                w_o: Array2::random((d, d), normal),
                b_o: Array1::zeros(d),
            };

            let ff = FeedForwardParams {
                w1: Array2::random((d, config.d_ff), normal),
                b1: Array1::zeros(config.d_ff),
                w2: Array2::random((config.d_ff, d), normal),
                b2: Array1::zeros(d),
            };

            blocks.push(TransformerBlockParams {
                attn,
                ff,
                ln1_gamma: Array1::ones(d),
                ln1_beta: Array1::zeros(d),
                ln2_gamma: Array1::ones(d),
                ln2_beta: Array1::zeros(d),
            });
        }

        Transformer {
            config: config.clone(),
            token_embedding,
            blocks,
            final_ln_gamma: Array1::ones(d),
            final_ln_beta: Array1::zeros(d),
            output_projection: Array2::random((d, config.vocab_size as usize), normal),
        }
    }

    /// Forward pass: tokens -> logits
    pub fn forward(&self, token_ids: &[u32]) -> Array2<f32> {
        let seq_len = token_ids.len().min(self.config.max_seq_len);
        let ids = &token_ids[..seq_len];
        let d = self.config.d_model;

        // Token embeddings + positional encoding
        let tok_emb = embedding_lookup(&self.token_embedding, ids);
        let pos_enc = positional_encoding(seq_len, d);
        let mut x = tok_emb + pos_enc;

        // Causal mask
        let mask = causal_mask(seq_len);

        // Pass through each transformer block
        for block in &self.blocks {
            x = self.transformer_block(&x, block, &mask);
        }

        // Final layer norm
        x = layer_norm(&x, &self.final_ln_gamma, &self.final_ln_beta, 1e-5);

        // Project to vocabulary
        matmul(&x, &self.output_projection)
    }

    /// Single transformer block: attention + feed-forward with residuals
    fn transformer_block(
        &self,
        x: &Array2<f32>,
        block: &TransformerBlockParams,
        mask: &Array2<f32>,
    ) -> Array2<f32> {
        // Pre-norm attention
        let normed = layer_norm(x, &block.ln1_gamma, &block.ln1_beta, 1e-5);
        let attn_out = self.multi_head_attention(&normed, &block.attn, mask);
        let x = x + &attn_out;

        // Pre-norm feed-forward
        let normed = layer_norm(&x, &block.ln2_gamma, &block.ln2_beta, 1e-5);
        let ff_out = self.feed_forward(&normed, &block.ff);
        &x + &ff_out
    }

    /// Multi-head self-attention
    fn multi_head_attention(
        &self,
        x: &Array2<f32>,
        params: &MultiHeadAttentionParams,
        mask: &Array2<f32>,
    ) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let d = self.config.d_model;
        let d_head = d / self.config.n_heads;
        let scale = (d_head as f32).sqrt();

        let mut head_outputs = Vec::new();

        for head in &params.heads {
            // Q, K, V projections
            let q = matmul(x, &head.w_q) + &head.b_q;
            let k = matmul(x, &head.w_k) + &head.b_k;
            let v = matmul(x, &head.w_v) + &head.b_v;

            // Attention scores: Q @ K^T / sqrt(d_head)
            let k_t = k.t().to_owned();
            let mut scores = matmul(&q, &k_t);
            scores /= scale;

            // Apply causal mask
            scores = scores + mask;

            // Softmax
            let weights = softmax_2d(&scores);

            // Weighted sum of values
            let out = matmul(&weights, &v);
            head_outputs.push(out);
        }

        // Concatenate heads
        let mut concat = Array2::<f32>::zeros((seq_len, d));
        for (i, head_out) in head_outputs.iter().enumerate() {
            concat
                .slice_mut(s![.., i * d_head..(i + 1) * d_head])
                .assign(head_out);
        }

        // Output projection
        matmul(&concat, &params.w_o) + &params.b_o
    }

    /// Feed-forward network: Linear -> GELU -> Linear
    fn feed_forward(&self, x: &Array2<f32>, params: &FeedForwardParams) -> Array2<f32> {
        let hidden = gelu(&(matmul(x, &params.w1) + &params.b1));
        matmul(&hidden, &params.w2) + &params.b2
    }

    /// Save model to a binary file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let encoded = bincode::serialize(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        std::fs::write(path, encoded)
    }

    /// Load model from a binary file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let model: Transformer = bincode::deserialize(&data).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        Ok(model)
    }

    /// Count total parameters
    pub fn total_params(&self) -> usize {
        self.config.param_count()
    }
}

/// Collect all mutable parameter references for the optimizer
pub fn collect_params(model: &mut Transformer) -> Vec<ParamRef> {
    let mut params = Vec::new();

    params.push(ParamRef::Array2(&mut model.token_embedding));

    for block in &mut model.blocks {
        for head in &mut block.attn.heads {
            params.push(ParamRef::Array2(&mut head.w_q));
            params.push(ParamRef::Array2(&mut head.w_k));
            params.push(ParamRef::Array2(&mut head.w_v));
            params.push(ParamRef::Array1(&mut head.b_q));
            params.push(ParamRef::Array1(&mut head.b_k));
            params.push(ParamRef::Array1(&mut head.b_v));
        }
        params.push(ParamRef::Array2(&mut block.attn.w_o));
        params.push(ParamRef::Array1(&mut block.attn.b_o));
        params.push(ParamRef::Array2(&mut block.ff.w1));
        params.push(ParamRef::Array1(&mut block.ff.b1));
        params.push(ParamRef::Array2(&mut block.ff.w2));
        params.push(ParamRef::Array1(&mut block.ff.b2));
        params.push(ParamRef::Array1(&mut block.ln1_gamma));
        params.push(ParamRef::Array1(&mut block.ln1_beta));
        params.push(ParamRef::Array1(&mut block.ln2_gamma));
        params.push(ParamRef::Array1(&mut block.ln2_beta));
    }

    params.push(ParamRef::Array1(&mut model.final_ln_gamma));
    params.push(ParamRef::Array1(&mut model.final_ln_beta));
    params.push(ParamRef::Array2(&mut model.output_projection));

    params
}

/// A reference to a parameter (1D or 2D array)
pub enum ParamRef<'a> {
    Array1(&'a mut Array1<f32>),
    Array2(&'a mut Array2<f32>),
}
