use crate::tensor::{Tensor, ToF32};
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use crate::gpu; // 引入 GPU kernel 实现

// get (row) vectors from a 2D table given a list of indices
#[cfg(feature = "gpu")]
pub fn gather<U: Copy + ToF32 + Default + cust::memory::DeviceCopy>(
    y: &mut Tensor<f32>,
    indices: &Tensor<u32>,
    table: &Tensor<U>,
) {
    // 调用 GPU 加速实现，gpu::gather_kernel 应该封装对 CUDA kernel 的调用
    crate::gpu::gather_kernel(y, indices, table).expect("GPU gather failed");
}

#[cfg(not(feature = "gpu"))]
pub fn gather<U: Copy + ToF32 + Default>(
    y: &mut Tensor<f32>,
    indices: &Tensor<u32>,
    table: &Tensor<U>,
) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    let y_data = unsafe { y.data_mut() };
    for i in 0..length {
        let idx = indices.data()[i] as usize;
        let src_start = idx * dim;
        let src = &table.data()[src_start..src_start + dim];
        let dst = &mut y_data[i * dim..(i + 1) * dim];
        for j in 0..dim {
            dst[j] = src[j].to_f32();
        }
    }
}

// RoPE: Rotary Positional Embedding
#[cfg(feature = "gpu")]
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    // 调用 GPU 实现，对应的 GPU kernel 应在 crate::gpu 模块中实现
    crate::gpu::rope_kernel(y, start_pos, theta).expect("GPU rope failed");
}

#[cfg(not(feature = "gpu"))]
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    // 原有 CPU 实现
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
#[cfg(feature = "gpu")]
pub fn masked_softmax(y: &mut Tensor<f32>) {
    // 调用 GPU 实现。如果需要，GPU 版本可以将 softmax 操作在 kernel 中并行化
    crate::gpu::masked_softmax_kernel(y).expect("GPU masked_softmax failed");
}

#[cfg(not(feature = "gpu"))]
pub fn masked_softmax(y: &mut Tensor<f32>) {
    // 原有 CPU 实现
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

#[cfg(feature = "gpu")]
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // 调用 GPU 实现，GPU 内核应完成归一化操作
    crate::gpu::rms_norm_kernel(y, x, w, epsilon).expect("GPU rms_norm failed");
}

#[cfg(not(feature = "gpu"))]
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // 原有 CPU 实现
    let x_shape = x.shape();
    assert!(x_shape.len() >= 1, "x 必须至少为1维的张量");
    let n = *x_shape.last().unwrap() as usize;

    assert!(y.size() == x.size(), "y 与 x 的元素数量必须一致");
    assert!(w.size() == n, "权重张量 w 的大小必须与最后一维一致");

    let total = x.size();
    let _batch = total / n;

    let x_data = x.data();
    let y_data = unsafe { y.data_mut() };
    let w_data = w.data();

    y_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(b, y_chunk)| {
            let x_chunk = &x_data[b * n..(b + 1) * n];
            let sum_sq: f32 = x_chunk.iter().map(|&v| v * v).sum();
            let norm_factor = ((sum_sq / n as f32) + epsilon).sqrt();
            for j in 0..n {
                y_chunk[j] = w_data[j] * x_chunk[j] / norm_factor;
            }
        });
}

// y = silu(x) * y
// hint: this is an element-wise operation
#[cfg(feature = "gpu")]
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // 调用 GPU 实现，GPU kernel 内部将完成 element-wise 计算
    crate::gpu::swiglu_kernel(y, x).expect("GPU swiglu failed");
}

#[cfg(not(feature = "gpu"))]
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // 原有 CPU 实现
    let len = y.size();
    assert!(len == x.size());
    let x_data = x.data();
    let y_data = unsafe { y.data_mut() };
    for i in 0..len {
        let xi = x_data[i];
        let sigmoid = 1.0 / (1.0 + (-xi).exp());
        let silu = xi * sigmoid;
        y_data[i] *= silu;
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
#[cfg(feature = "gpu")]
pub fn matmul_transb<U: Copy + ToF32 + Default + Sync + cust::memory::DeviceCopy>(
    c: &mut Tensor<f32>,
    beta: f32,
    a: &Tensor<f32>,
    b: &Tensor<U>,
    alpha: f32,
) {
    // 调用 GPU 实现，可以使用 cuBLAS 等库来完成矩阵乘法
    crate::gpu::matmul_transb_kernel(c, beta, a, b, alpha).expect("GPU matmul_transb failed");
}

#[cfg(not(feature = "gpu"))]
pub fn matmul_transb<U: Copy + ToF32 + Default + Sync>(
    c: &mut Tensor<f32>,
    beta: f32,
    a: &Tensor<f32>,
    b: &Tensor<U>,
    alpha: f32,
) {
    // 原有 CPU 实现
    let a_shape = a.shape();
    assert!(a_shape.len() == 2, "A 必须是二维矩阵");
    let m = a_shape[0];
    let k = a_shape[1];

    let b_shape = b.shape();
    assert!(b_shape.len() == 2, "B 必须是二维矩阵");
    assert!(b_shape[1] == k, "B 的列数必须与 A 的列数相等");
    let n = b_shape[0];

    let c_shape = c.shape();
    assert!(c_shape.len() == 2, "C 必须是二维矩阵");
    assert!(
        c_shape[0] == m && c_shape[1] == n,
        "C 的形状必须为 m×n"
    );

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    c_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[j * k + p].to_f32();
                }
                row[j] = beta * row[j] + alpha * sum;
            }
        });
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
