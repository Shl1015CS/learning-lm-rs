use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        println!("Available tensors: {:?}", safetensor.names());

        // 内部闭包：根据 key 从 safetensors 中加载张量，并转换为 Tensor<f32>
        // safetensors 中存储的是原始数据，每个元素占 4 字节，不需要再对张量做形变
        let load_tensor = |key: &str| -> Tensor<f32> {
            let view = safetensor
                .tensor(key)
                .unwrap_or_else(|_| panic!("未找到张量：{}", key));
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().expect("chunk 长度不足");
                    f32::from_le_bytes(bytes)
                })
                .collect();
            let shape = view.shape().to_vec();
            Tensor::new(f32_data, &shape)
        };

        // 根据配置决定 embedding 参数的加载 key
        // 当共享 embedding 时，safetensors 中只存储 lm_head.weight
        let embedding_table = if config.tie_word_embeddings {
            load_tensor("lm_head.weight")
        } else {
            load_tensor("model.embed_tokens.weight")
        };

        let n_layers = config.num_hidden_layers;
        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut q_proj_weights = Vec::with_capacity(n_layers);
        let mut k_proj_weights = Vec::with_capacity(n_layers);
        let mut v_proj_weights = Vec::with_capacity(n_layers);
        let mut o_proj_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut up_proj_weights = Vec::with_capacity(n_layers);
        let mut gate_proj_weights = Vec::with_capacity(n_layers);
        let mut down_proj_weights = Vec::with_capacity(n_layers);

        // 按照模型架构约定加载每一层的各项参数
        for i in 0..n_layers {
            let base = format!("model.layers.{}", i);
            attn_norm_weights.push(load_tensor(&format!("{}.input_layernorm.weight", base)));
            q_proj_weights.push(load_tensor(&format!("{}.self_attn.q_proj.weight", base)));
            k_proj_weights.push(load_tensor(&format!("{}.self_attn.k_proj.weight", base)));
            v_proj_weights.push(load_tensor(&format!("{}.self_attn.v_proj.weight", base)));
            o_proj_weights.push(load_tensor(&format!("{}.self_attn.o_proj.weight", base)));

            ffn_norm_weights.push(load_tensor(&format!("{}.post_attention_layernorm.weight", base)));
            up_proj_weights.push(load_tensor(&format!("{}.mlp.up_proj.weight", base)));
            gate_proj_weights.push(load_tensor(&format!("{}.mlp.gate_proj.weight", base)));
            down_proj_weights.push(load_tensor(&format!("{}.mlp.down_proj.weight", base)));
        }

        let rms_out_w = load_tensor("model.norm.weight");
        // 当共享 embedding 时，lm_head 与 embedding_table 数据相同
        let lm_head = if config.tie_word_embeddings {
            Tensor::new(embedding_table.data().to_vec(), &embedding_table.shape().to_vec())
        } else {
            load_tensor("lm_head.weight")
        };

        LLamaParams {
            embedding_table,
            rms_att_w: attn_norm_weights,
            wq: q_proj_weights,
            wk: k_proj_weights,
            wv: v_proj_weights,
            wo: o_proj_weights,
            rms_ffn_w: ffn_norm_weights,
            w_up: up_proj_weights,
            w_gate: gate_proj_weights,
            w_down: down_proj_weights,
            rms_out_w,
            lm_head,
        }
    }
}
