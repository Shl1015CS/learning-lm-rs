use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
use std::io::Write;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // 实现self_attention
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q.reshape(&vec![seq_len, self.n_q_h * self.dqkv]),
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            
            // 计算输出投影并更新残差
            let mut out_proj = Tensor::<f32>::default(&vec![seq_len, self.d]);
            OP::matmul_transb(&mut out_proj, 0., &hidden_states, &self.params.wo[layer], 1.0);
            let out_data = out_proj.data();
            let resid_data = unsafe { residual.data_mut() };
            for i in 0..resid_data.len() {
                resid_data[i] += out_data[i];
            }

            // 实现MLP部分
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // 将输入的 prompt 复制到结果中
        let mut result = token_ids.to_vec();
        
        // 如果未提供 prompt，则使用模型的 bos_token_id 作为初始 token
        if result.is_empty() {
            result.push(self.bos_token_id);
        }
    
        // 初始化一个 KV 缓存，将会在推理过程中被复用，避免重复分配内存
        let mut cache = self.new_cache();
    
        // 使用整个 prompt 对模型进行一次前向传播，更新 KV 缓存（不使用得到的 logits）
        let prompt_tensor = Tensor::<u32>::new(result.clone(), &vec![result.len()]);
        let _ = self.forward(&prompt_tensor, &mut cache);
    
        // 进入多轮推理，每一步只传入上一步生成的 token
        for _ in 0..max_len {
            // 取上一步生成的最后一个 token
            let last_token = *result.last().unwrap();
    
            // 构造只包含一个 token 的输入张量
            let input_tensor = Tensor::<u32>::new(vec![last_token], &vec![1]);
    
            // 前向传播：利用缓存中的历史信息得到下一个 token 的 logits
            let logits = self.forward(&input_tensor, &mut cache);
    
            // 使用采样算子根据 logits 采样出下一个 token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
    
            // 将新生成 token 添加到结果中
            result.push(next_token);
    
            // 如果生成结束符，则停止推理
            if next_token == self.eos_token_id {
                break;
            }
        }
    
        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let group_size = n_groups;
    // 将注意力分数计算封装到独立作用域
    {
        let att_data = unsafe { att_scores.data_mut() };
        let scale = 1.0 / (dqkv as f32).sqrt();
        
        // 每个KV头对应的Q头数量
        let group_size = n_groups;

        // 计算注意力分数
        for kv_head in 0..n_kv_h {
            for group in 0..group_size {
                for i in 0..seq_len {
                    for j in 0..total_seq_len {
                        let mut score = 0.0;
                        // 计算Q[i]和K[j]的点积
                        for p in 0..dqkv {
                            let q_idx = (i * n_kv_h * group_size + kv_head * group_size + group) * dqkv + p;
                            let k_idx = j * n_kv_h * dqkv + kv_head * dqkv + p;
                            score += q_data[q_idx] * k_data[k_idx];
                        }
                        let att_idx = kv_head * (group_size * seq_len * total_seq_len)
                            + group * (seq_len * total_seq_len)
                            + i * total_seq_len
                            + j;
                        att_data[att_idx] = score * scale;
                    }
                }
            }
        }
    } // 这里att_data离开作用域，释放可变借用

    // 现在可以安全地再次借用att_scores
    OP::masked_softmax(att_scores);

    // 后续代码需要重新获取可变引用
    let att_data = att_scores.data();
    let out_data = unsafe { hidden_states.data_mut() };
    
    // 计算注意力输出
    for kv_head in 0..n_kv_h {
        for group in 0..group_size {
            for i in 0..seq_len {
                let mut out_vec = [0.0; 128]; // 假设dqkv最大128
                for j in 0..total_seq_len {
                    let att_idx = kv_head * (group_size * seq_len * total_seq_len)
                        + group * (seq_len * total_seq_len)
                        + i * total_seq_len
                        + j;
                    let attn = att_data[att_idx];
                    
                    // 累加注意力权重到输出向量
                    for p in 0..dqkv {
                        let v_idx = j * n_kv_h * dqkv + kv_head * dqkv + p;
                        out_vec[p] += attn * v_data[v_idx];
                    }
                }
                
                // 将结果写入hidden_states缓冲区
                let out_idx = (i * n_kv_h * group_size + kv_head * group_size + group) * dqkv;
                for p in 0..dqkv {
                    out_data[out_idx + p] = out_vec[p];
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // 1. 通过 RMS normalization 计算 hidden = rms_norm(residual)
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    
    // 2. 计算 gate = hidden @ gate_weight.T
    //    注意：这里调用的是矩阵乘算子，设置 beta 为 0，alpha 为 1
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    
    // 3. 计算 up = hidden @ up_weight.T
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    
    // 4. 计算 SwiGLU 激活函数：act = gate * sigmoid(gate) * up
    //    我们使用 swiglu 算子实现：传入的参数会将 up 的每个元素乘以 gate * sigmoid(gate)
    //    执行后，up 中存储的就是 act 的结果
    OP::swiglu(up, gate);
    
    // 5. 计算 output = act @ down_weight.T
    //    输出 shape 为 [seq_len, d]，这里我们利用 hidden_states 这个缓冲区来存储 output
    OP::matmul_transb(hidden_states, 0.0, up, w_down, 1.0);
    
    // 6. 残差连接：更新 residual = output + residual
    let out_data = hidden_states.data();
    let size = residual.size();
    let resid_data = unsafe { residual.data_mut() };
    for i in 0..size {
        resid_data[i] += out_data[i];
    }
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
// impl<T> Tensor<T> 
// where
//     T: num_traits::Float + Default + rand::distributions::uniform::SampleUniform,
// {
//     pub fn random_normal(shape: &[usize], mean: T, std: T) -> Self {
//         use rand::prelude::*;
//         use rand_distr::Normal;
        

//         let size = shape.iter().product();
//         let mut rng = rand::thread_rng();
//         let normal = Normal::new(mean.to_f64().unwrap(), std.to_f64().unwrap()).unwrap();
        
//         let data: Vec<T> = (0..size)
//             .map(|_| T::from(normal.sample(&mut rng)).unwrap())
//             .collect();
        
//         // 将 shape 从 &[usize] 转换为 Vec<usize>，以满足 Tensor::new 的参数要求
//         let shape_vec = shape.to_vec();
//         Tensor::new(data, &shape_vec)
//     }
// }

// impl<T: Default + Clone + Copy> Tensor<T> {
//     pub fn zeros(shape: &[usize]) -> Self {
//         let size = shape.iter().product();
//         let data = vec![T::default(); size];
//         let shape_vec = shape.to_vec();
//         Tensor::new(data, &shape_vec)
//     }
// }

// #[test]
// fn test_self_attention() {
//     let dqkv = 16;      // 每个头的维度
//     let n_kv_h = 4;     // KV头数量
//     let n_groups = 2;    // 每个KV头对应的Q头组数
//     let seq_len = 1;     // 输入序列长度
//     let total_seq_len = 8; // 总序列长度（包含缓存）
//     let hidden_dim = n_kv_h * n_groups * dqkv;

//     let mut hidden_states = Tensor::<f32>::random_normal(&vec![seq_len, hidden_dim], 0.0, 0.1);
//     let mut att_scores = Tensor::<f32>::zeros(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    
//     // 构造Q/K/V的模拟参数
//     let q = Tensor::<f32>::random_normal(&vec![seq_len, n_kv_h * n_groups * dqkv], 0.0, 0.1);
//     let k = Tensor::<f32>::random_normal(&vec![total_seq_len, n_kv_h * dqkv], 0.0, 0.1);
//     let v = Tensor::<f32>::random_normal(&vec![total_seq_len, n_kv_h * dqkv], 0.0, 0.1);

//     // 执行 self_attention 计算
//     self_attention(
//         &mut hidden_states,
//         &mut att_scores,
//         &q,
//         &k,
//         &v,
//         n_kv_h,
//         n_groups,
//         seq_len,
//         total_seq_len,
//         dqkv,
//     );

//     // 验证输出形状
//     assert_eq!(hidden_states.shape(), &vec![seq_len, hidden_dim]);
    
//     // 计算统计指标
//     let output_data = hidden_states.data();
//     let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
//     let variance = output_data.iter()
//         .map(|x| (x - mean).powi(2))
//         .sum::<f32>() / output_data.len() as f32;
//     let std = variance.sqrt();

//     println!("实际均值: {:.6}, 标准差: {:.6}", mean, std);
    
//     assert!(
//         mean.abs() < 0.1,
//         "输出均值应在±0.1范围内，实际值：{:.6}", mean
//     );
//     assert!(
//         std > 0.01 && std < 0.2,
//         "输出标准差应在0.01-0.2范围内，实际值：{:.6}", std
//     );

//     let last_head_start = (n_kv_h - 1) * n_groups * dqkv;
//     let last_position = &output_data[last_head_start..last_head_start + dqkv];
//     assert!(
//         last_position.iter().any(|&x| x.abs() > 0.001),
//         "最后一个头的输出应存在有效值"
//     );
// }
/// 用于表示一条对话消息
pub struct Message {
    pub role: String,    // "user" 或 "assistant"
    pub content: String, // 消息内容
}

/// 用于封装对话会话，保存历史对话和 KVCache，不再依赖 tokenizers
pub struct ChatSession {
    pub model: Llama<f32>,
    pub conversation: Vec<Message>,
    pub cache: KVCache<f32>,
}

impl ChatSession {
    /// 创建一个新的对话会话，内部初始化 KVCache 和空的对话记录
    pub fn new(model: Llama<f32>) -> Self {
        let cache = model.new_cache();
        ChatSession {
            model,
            conversation: Vec::new(),
            cache,
        }
    }
    
    /// 添加用户消息到对话中
    pub fn add_user_message(&mut self, content: String) {
        self.conversation.push(Message {
            role: "user".to_string(),
            content,
        });
    }
    
    /// 构建对话的 prompt 字符串，符合对话模板：
    /// {% for message in messages %}
    /// <|im_start|>role
    /// content<|im_end|>
    /// {% endfor %}<|im_start|>assistant
    pub fn build_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.conversation {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// 使用传入的 prompt token 序列更新 KVCache，然后进行多轮生成，
    /// 返回生成的 token 序列。
    pub fn chat(
        &mut self,
        prompt_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // self.cache = self.model.new_cache();
        let prompt_tensor = Tensor::<u32>::new(prompt_ids.to_vec(), &vec![prompt_ids.len()]);
        let _ = self.model.forward(&prompt_tensor, &mut self.cache);
        let mut generated = Vec::<u32>::new();
        for _ in 0..max_len {
            let last_token = if !generated.is_empty() {
                *generated.last().unwrap()
            } else {
                *prompt_ids.last().unwrap_or(&self.model.eos_token_id)
            };
            
            let input_tensor = Tensor::<u32>::new(vec![last_token], &vec![1]);
            let logits = self.model.forward(&input_tensor, &mut self.cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            
            if next_token == self.model.eos_token_id {
                break;
            }
            
            generated.push(next_token);
            
            // 添加进度显示（新增）
            print!(".");
            std::io::stdout().flush().unwrap();
        }
        println!(); // 换行
        
        generated
    }
    
    /// 将生成的 assistant 回复添加到对话历史中
    pub fn add_assistant_message(&mut self, content: String) {
        self.conversation.push(Message {
            role: "assistant".to_string(),
            content,
        });
    }
}