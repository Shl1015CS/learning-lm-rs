use std::fs::File;
use std::vec;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use uuid::Uuid;
use serde_json;
use safetensors::SafeTensors;
use rayon::prelude::*;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
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
    #[allow(dead_code)]
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config_file = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();
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
            params,
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
        let mut result = token_ids.to_vec();
        
        if result.is_empty() {
            result.push(self.bos_token_id);
        }

        let mut cache = self.new_cache();
        let prompt_tensor = Tensor::<u32>::new(result.clone(), &vec![result.len()]);
        let _ = self.forward(&prompt_tensor, &mut cache);

        for _ in 0..max_len {
            let last_token = *result.last().unwrap();
            let input_tensor = Tensor::<u32>::new(vec![last_token], &vec![1]);
            let logits = self.forward(&input_tensor, &mut cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);

            if next_token == self.eos_token_id {
                break;
            }
        }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>,
    att_scores: &mut Tensor<f32>,
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let scale = 1.0 / (dqkv as f32).sqrt();
    
    // 计算注意力分数
    {
        let scores: Vec<_> = (0..n_kv_h).into_par_iter().map(|kv_head| {
            let mut head_scores = vec![0.0; n_groups * seq_len * total_seq_len];
            
            for group in 0..n_groups {
                for i in 0..seq_len {
                    let base_q = (i * n_kv_h * n_groups + kv_head * n_groups + group) * dqkv;
                    let base_k = kv_head * dqkv;
                    
                    for j in 0..total_seq_len {
                        let score: f32 = (0..dqkv)
                            .map(|p| q_data[base_q + p] * k_data[j * n_kv_h * dqkv + base_k + p])
                            .sum();
                            
                        let idx = group * (seq_len * total_seq_len)
                            + i * total_seq_len
                            + j;
                        head_scores[idx] = score * scale;
                    }
                }
            }
            head_scores
        }).collect();

        // 将计算结果写回 att_scores
        let att_data = unsafe { att_scores.data_mut() };
        for (h, head_scores) in scores.iter().enumerate() {
            for (i, &score) in head_scores.iter().enumerate() {
                att_data[h * n_groups * seq_len * total_seq_len + i] = score;
            }
        }
    }

    OP::masked_softmax(att_scores);

    // 计算注意力输出
    let att_data = att_scores.data();
    let outputs: Vec<_> = (0..n_kv_h).into_par_iter().map(|kv_head| {
        let mut head_output = vec![0.0; n_groups * seq_len * dqkv];
        
        for group in 0..n_groups {
            for i in 0..seq_len {
                let mut out_buf = vec![0.0; dqkv];
                
                let att_base = kv_head * (n_groups * seq_len * total_seq_len)
                    + group * (seq_len * total_seq_len)
                    + i * total_seq_len;
                    
                let v_base = kv_head * dqkv;
                
                for j in 0..total_seq_len {
                    let attn = att_data[att_base + j];
                    let v_offset = j * n_kv_h * dqkv + v_base;
                    
                    for p in 0..dqkv {
                        out_buf[p] += attn * v_data[v_offset + p];
                    }
                }
                
                let out_offset = group * seq_len * dqkv + i * dqkv;
                head_output[out_offset..out_offset + dqkv].copy_from_slice(&out_buf);
            }
        }
        head_output
    }).collect();

    // 将计算结果写回 hidden_states
    let out_data = unsafe { hidden_states.data_mut() };
    for (h, head_output) in outputs.iter().enumerate() {
        for group in 0..n_groups {
            for i in 0..seq_len {
                let src_offset = group * seq_len * dqkv + i * dqkv;
                let dst_offset = (i * n_kv_h * n_groups + h * n_groups + group) * dqkv;
                out_data[dst_offset..dst_offset + dqkv]
                    .copy_from_slice(&head_output[src_offset..src_offset + dqkv]);
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

/// 表示一条对话消息
#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,    // "user" 或 "assistant"
    pub content: String, // 消息内容
}

pub struct ChatSession {
    pub session_id: String,
    pub history: Vec<String>,
    pub model: Arc<Llama<f32>>,
    pub cache: KVCache<f32>,
}

impl ChatSession {
    pub fn new(session_id: String, model: Arc<Llama<f32>>) -> Self {
        let cache = model.new_cache();
        ChatSession {
            session_id,
            history: Vec::new(),
            model,
            cache,
        }
    }

    pub fn add_user_message(&mut self, content: String) {
        self.history.push(format!("User: {}", content));
        self.cache = self.model.new_cache();
    }

    pub fn add_assistant_message(&mut self, content: String) {
        self.history.push(format!("Assistant: {}", content));
        self.cache = self.model.new_cache();
    }

    pub fn build_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.history {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
    pub fn chat(
        &mut self,
        prompt_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let prompt_tensor = Tensor::<u32>::new(prompt_ids.to_vec(), &vec![prompt_ids.len()]);
        let _ = self.model.forward(&prompt_tensor, &mut self.cache);
        
        let mut generated = Vec::with_capacity(max_len);
        let mut input_tensor = Tensor::<u32>::new(vec![0], &vec![1]);
        
        for _ in 0..max_len {
            let last_token = generated.last()
                .map_or_else(
                    || prompt_ids.last().unwrap_or(&self.model.eos_token_id),
                    |t| t
                );
            
            unsafe {
                input_tensor.data_mut()[0] = *last_token;
            }
            let logits = self.model.forward(&input_tensor, &mut self.cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            
            if next_token == self.model.eos_token_id {
                break;
            }
            
            generated.push(next_token);
            print!(".");
            std::io::stdout().flush().unwrap();
        }
        println!();
        
        generated
    }

    pub fn get_history(&self) -> Vec<String> {
        self.history.clone()
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.cache = self.model.new_cache();
    }

    pub fn rollback(&mut self, version_index: usize) {
        if version_index < self.history.len() {
            self.history.truncate(version_index);
        }
    }
}
pub struct SessionManager {
    pub sessions: Mutex<HashMap<String, Arc<Mutex<ChatSession>>>>,
}

impl SessionManager {
    /// 创建新的 SessionManager 实例
    pub fn new() -> Self {
        SessionManager {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_session(&self, model: Arc<Llama<f32>>) -> String {
        let session_id = Uuid::new_v4().to_string();
        let session = ChatSession::new(session_id.clone(), model);
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.clone(), Arc::new(Mutex::new(session)));
        session_id
    }
    pub fn get_session(&self, session_id: &str) -> Option<Arc<Mutex<ChatSession>>> {
        self.sessions.lock().unwrap().get(session_id).cloned()
    }
    pub fn update_session(&self, session_id: &str, session: Arc<Mutex<ChatSession>>) {
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.to_string(), session);
    }

    pub fn rollback_session(&self, session_id: &str, version_index: usize) -> Option<()> {
        let sessions_lock = self.sessions.lock().unwrap();
        if let Some(session_arc) = sessions_lock.get(session_id) {
            let mut session = session_arc.lock().unwrap();
            session.rollback(version_index);
            Some(())
        } else {
            None
        }
    }

    pub fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.lock().unwrap();
        sessions.keys().cloned().collect()
    }

    pub fn get_or_create_session(&self, session_id: &str, model: Arc<Llama<f32>>) -> Arc<Mutex<ChatSession>> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(ChatSession::new(session_id.to_string(), model))))
            .clone()
    }
}