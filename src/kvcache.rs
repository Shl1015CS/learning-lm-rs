#![allow(dead_code)]

use std::{usize, vec};

use crate::tensor::Tensor;

/// KVCache 用于存储每一层的 key/value 缓存，
/// 每个缓存的维度为 (max_seq_len, n_kv_head * dqkv)。
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // 每层的 key 缓存
    v_cache: Vec<Tensor<T>>, // 每层的 value 缓存
    pub max_seq_len: usize,  // 最大序列长度
    pub dim: usize,          // 每个缓存的列数（n_kv_head * dqkv）
    length: usize,           // 当前缓存的序列长度
}

impl<T: Default + Copy> KVCache<T> {
    /// 创建新的 KVCache 实例
    ///
    /// 参数:
    /// - n_layers: 模型层数
    /// - max_seq_len: 最大序列长度
    /// - dim: 每层缓存数据列数（通常为 n_kv_head * dqkv）
    /// - init_len: 初始序列长度（一般为 0）
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    /// 返回指定层的 key 缓存片段，从 start 开始到当前缓存长度结束
    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// 返回指定层的 value 缓存片段，从 start 开始到当前缓存长度结束
    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// 增加当前缓存的序列长度
    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    /// 获取当前缓存的序列长度
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// 重置 KVCache，将所有层的缓存数据清空，并将序列长度重置为 0
    pub fn reset(&mut self) {
        for tensor in self.k_cache.iter_mut() {
            *tensor = Tensor::default(&vec![self.max_seq_len, self.dim]);
        }
        for tensor in self.v_cache.iter_mut() {
            *tensor = Tensor::default(&vec![self.max_seq_len, self.dim]);
        }
        self.length = 0;
    }
}
