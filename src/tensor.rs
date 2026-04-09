/// Lightweight tensor operations built on ndarray.
/// This module provides the math primitives the transformer needs:
/// matrix multiply, softmax, layer norm, GELU activation, etc.
///
/// When the `cuda` feature is enabled, matrix multiplications are
/// offloaded to the GPU via cuBLAS. Call `init_gpu()` at startup.
use ndarray::{Array1, Array2, Array3, Axis, s};

#[cfg(feature = "cuda")]
use std::cell::RefCell;

#[cfg(feature = "cuda")]
use crate::gpu::GpuContext;

#[cfg(feature = "cuda")]
thread_local! {
    static GPU: RefCell<Option<GpuContext>> = RefCell::new(None);
}

/// Initialize the GPU context for the current thread.
/// Call this once at the start of your program.
#[cfg(feature = "cuda")]
pub fn init_gpu() {
    GPU.with(|g| {
        let mut g = g.borrow_mut();
        if g.is_none() {
            match GpuContext::new() {
                Ok(ctx) => {
                    println!("[GPU] CUDA initialized successfully");
                    *g = Some(ctx);
                }
                Err(e) => {
                    eprintln!("[GPU] Failed to initialize CUDA: {} — falling back to CPU", e);
                }
            }
        }
    });
}

/// No-op when CUDA is not compiled in.
#[cfg(not(feature = "cuda"))]
pub fn init_gpu() {}

/// Matrix multiply: (M x K) @ (K x N) -> (M x N)
/// Uses cuBLAS on GPU when the `cuda` feature is enabled and GPU is initialized.
pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    #[cfg(feature = "cuda")]
    {
        let result = GPU.with(|g| {
            let g = g.borrow();
            g.as_ref().map(|gpu| gpu.matmul(a, b))
        });
        if let Some(c) = result {
            return c;
        }
    }
    a.dot(b)
}

/// Batched matrix multiply: (B x M x K) @ (B x K x N) -> (B x M x N)
pub fn batched_matmul(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    let batch = a.shape()[0];
    let m = a.shape()[1];
    let n = b.shape()[2];
    let mut out = Array3::<f32>::zeros((batch, m, n));
    for i in 0..batch {
        let a_slice = a.slice(s![i, .., ..]);
        let b_slice = b.slice(s![i, .., ..]);
        out.slice_mut(s![i, .., ..]).assign(&a_slice.dot(&b_slice));
    }
    out
}

/// Softmax along the last axis of a 2D array
pub fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let max_vals = x.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    });
    let shifted = x - &max_vals.insert_axis(Axis(1));
    let exp = shifted.mapv(f32::exp);
    let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
    &exp / &sum
}

/// Softmax along the last axis of a 3D array (batch x seq x seq)
pub fn softmax_3d(x: &Array3<f32>) -> Array3<f32> {
    let shape = x.shape().to_vec();
    let mut out = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    for b in 0..shape[0] {
        let slice = x.slice(s![b, .., ..]).to_owned();
        out.slice_mut(s![b, .., ..]).assign(&softmax_2d(&slice));
    }
    out
}

/// Layer normalization over the last axis
pub fn layer_norm(x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>, eps: f32) -> Array2<f32> {
    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let centered = x - &mean;
    let var = (&centered * &centered).mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let normed = &centered / &(var + eps).mapv(f32::sqrt);
    &normed * gamma + beta
}

/// GELU activation function (approximate)
pub fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Create a causal attention mask (upper triangle = -infinity)
pub fn causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::<f32>::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[i, j]] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Embedding lookup: given token IDs, return rows from the embedding matrix
pub fn embedding_lookup(table: &Array2<f32>, ids: &[u32]) -> Array2<f32> {
    let d = table.shape()[1];
    let mut out = Array2::<f32>::zeros((ids.len(), d));
    for (i, &id) in ids.iter().enumerate() {
        out.slice_mut(s![i, ..]).assign(&table.slice(s![id as usize, ..]));
    }
    out
}

/// Positional encoding (sinusoidal)
pub fn positional_encoding(seq_len: usize, d_model: usize) -> Array2<f32> {
    let mut pe = Array2::<f32>::zeros((seq_len, d_model));
    for pos in 0..seq_len {
        for i in 0..(d_model / 2) {
            let angle = pos as f32 / (10000.0_f32).powf(2.0 * i as f32 / d_model as f32);
            pe[[pos, 2 * i]] = angle.sin();
            pe[[pos, 2 * i + 1]] = angle.cos();
        }
    }
    pe
}

/// Cross-entropy loss between logits and target token IDs
pub fn cross_entropy_loss(logits: &Array2<f32>, targets: &[u32]) -> f32 {
    let probs = softmax_2d(logits);
    let mut loss = 0.0;
    for (i, &target) in targets.iter().enumerate() {
        let p = probs[[i, target as usize]].max(1e-10);
        loss -= p.ln();
    }
    loss / targets.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0]];
        let result = softmax_2d(&x);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a, &b);
        assert_eq!(c[[0, 0]], 19.0);
        assert_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_causal_mask() {
        let mask = causal_mask(3);
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[1, 1]], 0.0);
    }
}
