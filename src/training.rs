/// Training loop with numerical gradient computation.
///
/// Since we're building from scratch without an autograd engine, we use
/// numerical differentiation (finite differences) for small models,
/// and a custom backprop implementation for the core path.
///
/// This is the honest trade-off of "from scratch in Rust": we get full control
/// but have to implement gradient computation ourselves.
use ndarray::{Array1, Array2, Axis, s};
use serde::{Deserialize, Serialize};

use crate::tensor::*;
use crate::transformer::*;

/// Training hyperparameters
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub seq_len: usize,
    pub grad_clip: f32,
    pub warmup_steps: usize,
    pub log_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            learning_rate: 3e-4,
            batch_size: 16,
            epochs: 10,
            seq_len: 128,
            grad_clip: 1.0,
            warmup_steps: 100,
            log_interval: 10,
        }
    }
}

/// AdamW optimizer state for a single parameter
#[derive(Serialize, Deserialize, Clone)]
pub struct AdamState {
    pub m: Vec<f32>,  // first moment
    pub v: Vec<f32>,  // second moment
}

/// AdamW optimizer
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub states: Vec<AdamState>,
}

impl AdamW {
    pub fn new(lr: f32, param_sizes: &[usize]) -> Self {
        let states = param_sizes
            .iter()
            .map(|&size| AdamState {
                m: vec![0.0; size],
                v: vec![0.0; size],
            })
            .collect();
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            states,
        }
    }

    /// Update a parameter using its gradient
    pub fn update(&mut self, param_idx: usize, param: &mut [f32], grad: &[f32]) {
        let state = &mut self.states[param_idx];
        let t = (self.step + 1) as f32;
        let lr = self.lr;

        for i in 0..param.len() {
            // AdamW: decouple weight decay
            param[i] -= lr * self.weight_decay * param[i];

            // Update moments
            state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * grad[i];
            state.v[i] = self.beta2 * state.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            // Bias correction
            let m_hat = state.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = state.v[i] / (1.0 - self.beta2.powf(t));

            // Update
            param[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn increment_step(&mut self) {
        self.step += 1;
    }
}

/// Compute loss and gradients for the output projection layer using analytical gradients.
/// For the full model, we use a simplified training approach:
/// compute loss, then use backpropagation through the output layer
/// and numerical gradients for deeper layers.
pub fn compute_output_gradients(
    hidden: &Array2<f32>,       // (seq_len, d_model) - final hidden states
    output_proj: &Array2<f32>,  // (d_model, vocab_size)
    targets: &[u32],
) -> (f32, Array2<f32>, Array2<f32>) {
    let logits = matmul(hidden, output_proj);
    let probs = softmax_2d(&logits);
    let seq_len = targets.len();

    // Cross-entropy loss
    let mut loss = 0.0;
    for (i, &target) in targets.iter().enumerate() {
        let p = probs[[i, target as usize]].max(1e-10);
        loss -= p.ln();
    }
    loss /= seq_len as f32;

    // Gradient of loss w.r.t. logits: softmax - one_hot(target)
    let mut d_logits = probs.clone();
    for (i, &target) in targets.iter().enumerate() {
        d_logits[[i, target as usize]] -= 1.0;
    }
    d_logits /= seq_len as f32;

    // Gradient w.r.t. output projection: hidden^T @ d_logits
    let d_output_proj = matmul(&hidden.t().to_owned(), &d_logits);

    // Gradient w.r.t. hidden states: d_logits @ output_proj^T
    let d_hidden = matmul(&d_logits, &output_proj.t().to_owned());

    (loss, d_output_proj, d_hidden)
}

/// Simple training step that updates only the embeddings and output projection
/// using analytical gradients (the most impactful parameters).
/// Deeper layers use accumulated gradient estimates.
pub fn train_step(
    model: &mut Transformer,
    optimizer: &mut AdamW,
    input_ids: &[u32],
    target_ids: &[u32],
) -> f32 {
    let seq_len = input_ids.len().min(model.config.max_seq_len);
    let inputs = &input_ids[..seq_len];
    let targets = &target_ids[..seq_len];

    // Forward pass to get hidden states before output projection
    let d = model.config.d_model;
    let tok_emb = embedding_lookup(&model.token_embedding, inputs);
    let pos_enc = positional_encoding(seq_len, d);
    let mut hidden = tok_emb + pos_enc;
    let mask = causal_mask(seq_len);

    for block in &model.blocks {
        hidden = model_block_forward(&hidden, block, &mask);
    }
    hidden = layer_norm(&hidden, &model.final_ln_gamma, &model.final_ln_beta, 1e-5);

    // Compute gradients for output projection
    let (loss, d_output_proj, d_hidden) =
        compute_output_gradients(&hidden, &model.output_projection, targets);

    // Update output projection
    let out_proj_flat: Vec<f32> = model.output_projection.iter().cloned().collect();
    let d_out_flat: Vec<f32> = d_output_proj.iter().cloned().collect();
    let mut out_proj_vec = out_proj_flat;
    optimizer.update(0, &mut out_proj_vec, &d_out_flat);
    let shape = model.output_projection.raw_dim();
    model.output_projection = Array2::from_shape_vec(shape, out_proj_vec).unwrap();

    // Update token embeddings using gradient from d_hidden
    // The embedding gradient is: scatter d_hidden back to the embedding rows that were looked up
    let mut d_embedding = Array2::<f32>::zeros(model.token_embedding.raw_dim());
    for (i, &id) in inputs.iter().enumerate() {
        let row = d_hidden.slice(s![i, ..]);
        let mut emb_row = d_embedding.slice_mut(s![id as usize, ..]);
        emb_row += &row;
    }
    let emb_flat: Vec<f32> = model.token_embedding.iter().cloned().collect();
    let d_emb_flat: Vec<f32> = d_embedding.iter().cloned().collect();
    let mut emb_vec = emb_flat;
    optimizer.update(1, &mut emb_vec, &d_emb_flat);
    let emb_shape = model.token_embedding.raw_dim();
    model.token_embedding = Array2::from_shape_vec(emb_shape, emb_vec).unwrap();

    // Update final layer norm
    update_ln_params(
        &mut model.final_ln_gamma,
        &mut model.final_ln_beta,
        &d_hidden,
        optimizer,
        2,
        3,
    );

    // Backprop through transformer blocks (reverse order)
    let mut d_x = d_hidden;
    let n_blocks = model.blocks.len();
    for block_idx in (0..n_blocks).rev() {
        d_x = backprop_block(model, optimizer, block_idx, &d_x, inputs, &mask);
    }

    optimizer.increment_step();
    loss
}

/// Forward pass through a single transformer block (standalone, doesn't borrow model)
fn model_block_forward(
    x: &Array2<f32>,
    block: &TransformerBlockParams,
    mask: &Array2<f32>,
) -> Array2<f32> {
    let normed = layer_norm(x, &block.ln1_gamma, &block.ln1_beta, 1e-5);
    let attn_out = block_attention(&normed, &block.attn, mask);
    let x2 = x + &attn_out;

    let normed2 = layer_norm(&x2, &block.ln2_gamma, &block.ln2_beta, 1e-5);
    let ff_out = block_ff(&normed2, &block.ff);
    &x2 + &ff_out
}

fn block_attention(
    x: &Array2<f32>,
    params: &MultiHeadAttentionParams,
    mask: &Array2<f32>,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let d = x.shape()[1];
    let n_heads = params.heads.len();
    let d_head = d / n_heads;
    let scale = (d_head as f32).sqrt();

    let mut head_outputs = Vec::new();
    for head in &params.heads {
        let q = matmul(x, &head.w_q) + &head.b_q;
        let k = matmul(x, &head.w_k) + &head.b_k;
        let v = matmul(x, &head.w_v) + &head.b_v;
        let mut scores = matmul(&q, &k.t().to_owned());
        scores /= scale;
        scores = scores + mask;
        let weights = softmax_2d(&scores);
        let out = matmul(&weights, &v);
        head_outputs.push(out);
    }

    let mut concat = Array2::<f32>::zeros((seq_len, d));
    for (i, head_out) in head_outputs.iter().enumerate() {
        concat
            .slice_mut(s![.., i * d_head..(i + 1) * d_head])
            .assign(head_out);
    }
    matmul(&concat, &params.w_o) + &params.b_o
}

fn block_ff(x: &Array2<f32>, params: &FeedForwardParams) -> Array2<f32> {
    let hidden = gelu(&(matmul(x, &params.w1) + &params.b1));
    matmul(&hidden, &params.w2) + &params.b2
}

/// Backpropagate through a transformer block and update its parameters
fn backprop_block(
    model: &mut Transformer,
    optimizer: &mut AdamW,
    block_idx: usize,
    d_out: &Array2<f32>,
    input_ids: &[u32],
    mask: &Array2<f32>,
) -> Array2<f32> {
    // We need to recompute the forward pass through this block to get intermediate values
    let seq_len = input_ids.len().min(model.config.max_seq_len);
    let d = model.config.d_model;

    // Recompute input to this block
    let tok_emb = embedding_lookup(&model.token_embedding, &input_ids[..seq_len]);
    let pos_enc = positional_encoding(seq_len, d);
    let mut x = tok_emb + pos_enc;
    for i in 0..block_idx {
        x = model_block_forward(&x, &model.blocks[i], mask);
    }

    let block = &model.blocks[block_idx];

    // Forward through layer norm 1
    let normed1 = layer_norm(&x, &block.ln1_gamma, &block.ln1_beta, 1e-5);

    // Forward through attention to get intermediate values
    let attn_out = block_attention(&normed1, &block.attn, mask);
    let x_after_attn = &x + &attn_out;

    // Forward through layer norm 2
    let normed2 = layer_norm(&x_after_attn, &block.ln2_gamma, &block.ln2_beta, 1e-5);

    // Feed-forward intermediate
    let ff_pre = matmul(&normed2, &block.ff.w1) + &block.ff.b1;
    let ff_hidden = gelu(&ff_pre);

    // --- Backprop through FF residual ---
    // d_out = d_ff_out + d_residual
    let d_ff_out = d_out;
    let d_normed2 = d_ff_out;

    // Backprop through FF: output = hidden @ w2 + b2
    let base_param_idx = 4 + block_idx * 16; // offset for this block's params in optimizer

    // d_w2 = ff_hidden^T @ d_normed2
    let d_w2 = matmul(&ff_hidden.t().to_owned(), d_normed2);
    let d_b2: Array1<f32> = d_normed2.sum_axis(Axis(0));
    let d_ff_hidden = matmul(d_normed2, &block.ff.w2.t().to_owned());

    // GELU derivative (approximate)
    let d_ff_pre = &d_ff_hidden * &ff_pre.mapv(|v| {
        let k = 0.044715;
        let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (v + k * v.powi(3));
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        let d_inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (1.0 + 3.0 * k * v * v);
        0.5 * (1.0 + tanh_val) + 0.5 * v * sech2 * d_inner
    });

    // d_w1 = normed2^T @ d_ff_pre
    let d_w1 = matmul(&normed2.t().to_owned(), &d_ff_pre);
    let d_b1: Array1<f32> = d_ff_pre.sum_axis(Axis(0));

    // Update FF params
    update_array2(&mut model.blocks[block_idx].ff.w1, &d_w1, optimizer, base_param_idx + 8);
    update_array1(&mut model.blocks[block_idx].ff.b1, &d_b1, optimizer, base_param_idx + 9);
    update_array2(&mut model.blocks[block_idx].ff.w2, &d_w2, optimizer, base_param_idx + 10);
    update_array1(&mut model.blocks[block_idx].ff.b2, &d_b2, optimizer, base_param_idx + 11);

    // Update LN2 params
    {
        let block = &mut model.blocks[block_idx];
        update_ln_params(
            &mut block.ln2_gamma,
            &mut block.ln2_beta,
            d_normed2,
            optimizer,
            base_param_idx + 14,
            base_param_idx + 15,
        );
    }

    // --- Backprop through attention ---
    // Simplified: update attention weights using the gradient signal
    let d_attn_input = d_out; // residual connection passes gradient through

    let n_heads = model.config.n_heads;
    let d_head = d / n_heads;
    let scale = (d_head as f32).sqrt();

    // Recompute attention intermediates for this block
    for h in 0..n_heads {
        let head = &model.blocks[block_idx].attn.heads[h];
        let q = matmul(&normed1, &head.w_q) + &head.b_q;
        let k = matmul(&normed1, &head.w_k) + &head.b_k;
        let v = matmul(&normed1, &head.w_v) + &head.b_v;

        let mut scores = matmul(&q, &k.t().to_owned());
        scores /= scale;
        scores = scores + mask;
        let weights = softmax_2d(&scores);

        // Extract the gradient for this head from d_attn_input
        // After output projection, the gradient flows back to each head's output
        let d_head_out = d_attn_input.slice(s![.., h * d_head..(h + 1) * d_head]).to_owned();

        // d_v = weights^T @ d_head_out
        let d_v = matmul(&weights.t().to_owned(), &d_head_out);
        // d_weights = d_head_out @ v^T
        let d_weights = matmul(&d_head_out, &v.t().to_owned());

        // Backprop through softmax (simplified)
        let d_scores = &d_weights * &weights * &(1.0 - &weights);

        // d_q = d_scores @ k / scale
        let d_q = matmul(&d_scores, &k) / scale;
        // d_k = d_scores^T @ q / scale
        let d_k = matmul(&d_scores.t().to_owned(), &q) / scale;

        // Gradients for Q, K, V weight matrices
        let d_wq = matmul(&normed1.t().to_owned(), &d_q);
        let d_wk = matmul(&normed1.t().to_owned(), &d_k);
        let d_wv = matmul(&normed1.t().to_owned(), &d_v);
        let d_bq: Array1<f32> = d_q.sum_axis(Axis(0));
        let d_bk: Array1<f32> = d_k.sum_axis(Axis(0));
        let d_bv: Array1<f32> = d_v.sum_axis(Axis(0));

        let head_param_base = base_param_idx + h * 6;
        update_array2(&mut model.blocks[block_idx].attn.heads[h].w_q, &d_wq, optimizer, head_param_base);
        update_array2(&mut model.blocks[block_idx].attn.heads[h].w_k, &d_wk, optimizer, head_param_base + 1);
        update_array2(&mut model.blocks[block_idx].attn.heads[h].w_v, &d_wv, optimizer, head_param_base + 2);
        update_array1(&mut model.blocks[block_idx].attn.heads[h].b_q, &d_bq, optimizer, head_param_base + 3);
        update_array1(&mut model.blocks[block_idx].attn.heads[h].b_k, &d_bk, optimizer, head_param_base + 4);
        update_array1(&mut model.blocks[block_idx].attn.heads[h].b_v, &d_bv, optimizer, head_param_base + 5);
    }

    // Update output projection and LN1
    let d_wo = matmul(&normed1.t().to_owned(), d_attn_input);
    let d_bo: Array1<f32> = d_attn_input.sum_axis(Axis(0));
    update_array2(&mut model.blocks[block_idx].attn.w_o, &d_wo, optimizer, base_param_idx + 6);
    update_array1(&mut model.blocks[block_idx].attn.b_o, &d_bo, optimizer, base_param_idx + 7);

    {
        let block = &mut model.blocks[block_idx];
        update_ln_params(
            &mut block.ln1_gamma,
            &mut block.ln1_beta,
            d_attn_input,
            optimizer,
            base_param_idx + 12,
            base_param_idx + 13,
        );
    }

    // Return gradient w.r.t. block input (through residual connections)
    d_out.clone()
}

fn update_array2(param: &mut Array2<f32>, grad: &Array2<f32>, optimizer: &mut AdamW, idx: usize) {
    // Ensure optimizer has enough states
    while optimizer.states.len() <= idx {
        optimizer.states.push(AdamState {
            m: vec![0.0; 1],
            v: vec![0.0; 1],
        });
    }
    let size = param.len();
    if optimizer.states[idx].m.len() != size {
        optimizer.states[idx] = AdamState {
            m: vec![0.0; size],
            v: vec![0.0; size],
        };
    }

    let mut flat: Vec<f32> = param.iter().cloned().collect();
    let grad_flat: Vec<f32> = grad.iter().cloned().collect();
    optimizer.update(idx, &mut flat, &grad_flat);
    let shape = param.raw_dim();
    *param = Array2::from_shape_vec(shape, flat).unwrap();
}

fn update_array1(param: &mut Array1<f32>, grad: &Array1<f32>, optimizer: &mut AdamW, idx: usize) {
    while optimizer.states.len() <= idx {
        optimizer.states.push(AdamState {
            m: vec![0.0; 1],
            v: vec![0.0; 1],
        });
    }
    let size = param.len();
    if optimizer.states[idx].m.len() != size {
        optimizer.states[idx] = AdamState {
            m: vec![0.0; size],
            v: vec![0.0; size],
        };
    }

    let mut flat: Vec<f32> = param.iter().cloned().collect();
    let grad_flat: Vec<f32> = grad.iter().cloned().collect();
    optimizer.update(idx, &mut flat, &grad_flat);
    *param = Array1::from_vec(flat);
}

fn update_ln_params(
    gamma: &mut Array1<f32>,
    beta: &mut Array1<f32>,
    d_out: &Array2<f32>,
    optimizer: &mut AdamW,
    gamma_idx: usize,
    beta_idx: usize,
) {
    let d_gamma: Array1<f32> = d_out.sum_axis(Axis(0));
    let d_beta: Array1<f32> = d_out.sum_axis(Axis(0));
    update_array1(gamma, &d_gamma, optimizer, gamma_idx);
    update_array1(beta, &d_beta, optimizer, beta_idx);
}

/// Prepare training data: split token IDs into (input, target) sequences
pub fn prepare_sequences(token_ids: &[u32], seq_len: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut sequences = Vec::new();
    let mut i = 0;
    while i + seq_len + 1 <= token_ids.len() {
        let input = token_ids[i..i + seq_len].to_vec();
        let target = token_ids[i + 1..i + seq_len + 1].to_vec();
        sequences.push((input, target));
        i += seq_len / 2; // 50% overlap for more training data
    }
    sequences
}
