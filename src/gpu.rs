/// GPU acceleration via CUDA + cuBLAS.
///
/// This module provides GPU-accelerated matrix multiplication using cuBLAS,
/// which is the main bottleneck in transformer training and inference.
/// All other ops (softmax, layer norm, GELU) remain on CPU since they are
/// memory-bound rather than compute-bound at our scale.
use cudarc::cublas::safe::{CudaBlas, GemmConfig};
use cudarc::cublas::Gemm;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext, CudaStream};
use ndarray::Array2;
use std::sync::Arc;

/// Holds the CUDA context, stream, and cuBLAS handle for the session.
pub struct GpuContext {
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
}

impl GpuContext {
    /// Initialize CUDA on device 0 (the 3080 Ti).
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        Ok(GpuContext { stream, blas })
    }

    /// GPU-accelerated matrix multiply: (M x K) @ (K x N) -> (M x N)
    ///
    /// cuBLAS expects column-major layout, but ndarray is row-major.
    /// We use the identity: C = A @ B  <==>  C^T = B^T @ A^T
    /// Since row-major A is column-major A^T, we pass B then A to cuBLAS
    /// and get C in row-major order directly.
    pub fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let m = a.shape()[0]; // rows of A
        let k = a.shape()[1]; // cols of A = rows of B
        let n = b.shape()[1]; // cols of B

        assert_eq!(b.shape()[0], k, "matmul dimension mismatch");

        // Ensure contiguous layout
        let a_data = a.as_standard_layout();
        let b_data = b.as_standard_layout();

        let a_slice = a_data.as_slice().expect("a must be contiguous");
        let b_slice = b_data.as_slice().expect("b must be contiguous");

        // Upload to GPU
        let d_a = self.stream.clone_htod(a_slice).unwrap();
        let d_b = self.stream.clone_htod(b_slice).unwrap();
        let mut d_c = self.stream.alloc_zeros::<f32>(m * n).unwrap();

        // cuBLAS SGEMM in column-major: C^T = B^T @ A^T
        // Row-major C(m,n) = A(m,k) @ B(k,n)
        // In col-major terms: C_col(n,m) = B_col(n,k) @ A_col(k,m)
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,   // rows of op(B_col) = n
                        n: m as i32,   // cols of op(A_col) = m
                        k: k as i32,   // shared dimension
                        alpha: 1.0f32,
                        lda: n as i32, // leading dim of B in col-major = n
                        ldb: k as i32, // leading dim of A in col-major = k
                        beta: 0.0f32,
                        ldc: n as i32, // leading dim of C in col-major = n
                    },
                    &d_b, // B (first operand in col-major trick)
                    &d_a, // A (second operand)
                    &mut d_c,
                )
                .unwrap();
        }

        // Download result
        let c_host = self.stream.clone_dtoh(&d_c).unwrap();
        Array2::from_shape_vec((m, n), c_host).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpu_matmul() {
        let gpu = GpuContext::new().expect("CUDA init failed");
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];
        let c = gpu.matmul(&a, &b);
        assert!((c[[0, 0]] - 19.0).abs() < 1e-3);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-3);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-3);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_gpu_matmul_nonsquare() {
        let gpu = GpuContext::new().expect("CUDA init failed");
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]]; // 3x2
        let c = gpu.matmul(&a, &b); // should be 2x2
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c[[0, 0]] - 58.0).abs() < 1e-3);
        assert!((c[[0, 1]] - 64.0).abs() < 1e-3);
        assert!((c[[1, 0]] - 139.0).abs() < 1e-3);
        assert!((c[[1, 1]] - 154.0).abs() < 1e-3);
    }
}
