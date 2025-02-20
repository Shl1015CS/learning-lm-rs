use std::time::Instant;
use rand::Rng;
use crate::tensor::Tensor;
use crate::tensor::ToF32;
use crate::operators::{matmul_transb, rms_norm};

/// 串行实现的矩阵乘法：C = beta * C + alpha * A @ B^T
pub fn matmul_transb_serial<U: Copy + ToF32 + Default>(
    c: &mut Tensor<f32>,
    beta: f32,
    a: &Tensor<f32>,
    b: &Tensor<U>,
    alpha: f32,
) {
    let a_shape = a.shape();
    let m = a_shape[0];
    let k = a_shape[1];

    let b_shape = b.shape();
    let n = b_shape[0];

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[j * k + p].to_f32();
            }
            c_data[i * n + j] = beta * c_data[i * n + j] + alpha * sum;
        }
    }
}

/// 串行实现的 RMS 归一化：对每个向量进行归一化
pub fn rms_norm_serial(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let x_shape = x.shape();
    let n = *x_shape.last().unwrap() as usize;
    let total = x.size();
    let batch = total / n;

    let x_data = x.data();
    let y_data = unsafe { y.data_mut() };
    let w_data = w.data();

    for b in 0..batch {
        let offset = b * n;
        let mut sum_sq = 0.0;
        for j in 0..n {
            let val = x_data[offset + j];
            sum_sq += val * val;
        }
        let norm_factor = ((sum_sq / n as f32) + epsilon).sqrt();
        for j in 0..n {
            y_data[offset + j] = w_data[j] * x_data[offset + j] / norm_factor;
        }
    }
}

/// 根据给定的 shape 生成随机 Tensor<f32>
fn random_tensor(shape: &[usize]) -> Tensor<f32> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..size).map(|_| rng.gen_range(0.0..1.0)).collect();
    Tensor::new(data, &shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{matmul_transb, rms_norm};

    #[test]
    fn bench_matmul_transb() {
        // 设置较大的矩阵尺寸
        let m = 200;
        let k = 256;
        let n = 128;

        // 初始化 a, b, 和 c 张量
        let a = random_tensor(&[m, k]);
        let b = random_tensor(&[n, k]);
        let mut c_parallel = Tensor::<f32>::default(&vec![m, n]);
        let mut c_serial = Tensor::<f32>::default(&vec![m, n]);

        // 预热多次运行
        for _ in 0..10 {
            let _ = matmul_transb(&mut c_parallel, 0.0, &a, &b, 1.0);
            matmul_transb_serial(&mut c_serial, 0.0, &a, &b, 1.0);
        }

        // Benchmark 并行版本：调用 100 次
        let start = Instant::now();
        for _ in 0..100 {
            let _ = matmul_transb(&mut c_parallel, 0.0, &a, &b, 1.0);
        }
        let duration_parallel = start.elapsed();

        // Benchmark 串行版本：调用 100 次
        let start = Instant::now();
        for _ in 0..100 {
            matmul_transb_serial(&mut c_serial, 0.0, &a, &b, 1.0);
        }
        let duration_serial = start.elapsed();

        println!("matmul_transb_parallel: {:?} for 100 runs", duration_parallel);
        println!("matmul_transb_serial:   {:?} for 100 runs", duration_serial);
    }

    #[test]
    fn bench_rms_norm() {
        // 设置 tensor 尺寸，例如 1000 个向量，每个向量长度为 512
        let batch = 1000;
        let n = 512;
        let shape = vec![batch, n];

        let x = random_tensor(&shape);
        let mut y_parallel = Tensor::<f32>::default(&shape);
        let mut y_serial = Tensor::<f32>::default(&shape);
        let w = random_tensor(&[n]);

        // 预热多次运行
        for _ in 0..10 {
            rms_norm(&mut y_parallel, &x, &w, 1e-6);
            rms_norm_serial(&mut y_serial, &x, &w, 1e-6);
        }

        // Benchmark 并行版本：调用 100 次
        let start = Instant::now();
        for _ in 0..100 {
            rms_norm(&mut y_parallel, &x, &w, 1e-6);
        }
        let duration_parallel = start.elapsed();

        // Benchmark 串行版本：调用 100 次
        let start = Instant::now();
        for _ in 0..100 {
            rms_norm_serial(&mut y_serial, &x, &w, 1e-6);
        }
        let duration_serial = start.elapsed();

        println!("rms_norm_parallel: {:?} for 100 runs", duration_parallel);
        println!("rms_norm_serial:   {:?} for 100 runs", duration_serial);
    }
} 