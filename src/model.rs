

// use std::fs::File;
// use std::vec;

// use crate::config::LlamaConfigJson;
// use crate::kvcache::KVCache;
// use crate::operators as OP;
// use crate::params::LLamaParams;
// use crate::tensor::Tensor;
// use safetensors::SafeTensors;
// use std::path::Path;
// pub struct Llama<T> {
//     vocab: usize,           // vocab size
//     n_layers: usize,        // number of layers
//     n_q_h: usize,           // number of heads for q

//     n_kv_h: usize,          // number of heads for k and v
//     d: usize,               // dimension of hidden states
//     dqkv: usize,            // length of a single q, k, or v vector
//     di: usize,              // dimension of intermediate states
//     eps: f32,               // epsilon for RMS normalization
//     rope_theta: f32,        // rope theta for rope initialization
//     max_seq_len: usize,     // maximum sequence length
//     params: LLamaParams<T>, // trained weights of this model
//     bos_token_id: u32,      // start token id
//     eos_token_id: u32,      // end token id
// }

// impl Llama<f32> {
//     pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
//         let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
//         let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
//         let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
//         let safetensor = SafeTensors::deserialize(&model_file).unwrap();
//         let params = LLamaParams::from_safetensors(&safetensor, &config);

//         Self {
//             vocab: config.vocab_size,
//             n_layers: config.num_hidden_layers,
//             n_q_h: config.num_attention_heads,
//             n_kv_h: config.num_key_value_heads,
//             d: config.hidden_size,
//             dqkv: config.hidden_size / config.num_attention_heads,
//             di: config.intermediate_size,
//             eps: config.rms_norm_eps,
//             rope_theta: config.rope_theta,
//             max_seq_len: config.max_position_embeddings,
//             params: params,
//             bos_token_id: config.bos_token_id,
//             eos_token_id: config.eos_token_id,
//         }
//     }

//     pub fn new_cache(&self) -> KVCache<f32> {
//         KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
//     }

//     pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
//         let seq_len = input.size();
//         let past_seq_len = cache.len();
//         cache.increment(seq_len);
//         let total_seq_len = past_seq_len + seq_len;
//         let n_groups = self.n_q_h / self.n_kv_h;

//         // Some pre-allocated buffers that will be reused
//         let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
//         let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
//         let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
//         let mut att_scores =
//             Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
//         let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
//         let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

//         // Computation Starts Here
//         // Embedding lookup
//         OP::gather(&mut residual, input, &self.params.embedding_table);

//         for layer in 0..self.n_layers {
//             OP::rms_norm(
//                 &mut hidden_states,
//                 &residual,
//                 &self.params.rms_att_w[layer],
//                 self.eps,
//             );

//             let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
//             let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
//             let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
//             OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
//             OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
//             OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
//             OP::rope(
//                 q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
//                 past_seq_len,
//                 self.rope_theta,
//             );
//             OP::rope(
//                 k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
//                 past_seq_len,
//                 self.rope_theta,
//             );

//             let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
//             let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

//             todo!("self_attention(...)");
//             todo!("down_proj matmul and add residual");

//             todo!("mlp(...)");
//         }

//         // No matter what seq_len, the output is always a 1D vector of length vocab,
//         // which contains the probabilities for the next token.
//         let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
//         let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
//         let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

//         OP::rms_norm(
//             &mut hidden_states,
//             &residual,
//             &self.params.rms_out_w,
//             self.eps,
//         );

//         OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

//         logits
//     }



    

//     pub fn generate(
//         &self,
//         token_ids: &[u32],
//         max_len: usize,
//         top_p: f32,
//         top_k: u32,
//         temperature: f32,
//     ) -> Vec<u32>{
//         let mut result = Vec::<u32>::new();
        
//         todo!("实现文本生成");
        
//         result
//     }
// }

// fn self_attention(
//     hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
//     att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
//     q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
//     k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     n_kv_h: usize,
//     n_groups: usize,
//     seq_len: usize,
//     total_seq_len: usize,
//     dqkv: usize,
// ) {
//     todo!("Implement self_attention");
// }

// /*########################################################################################################################### */
// pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
//     // Ensure that y, x, and w have compatible shapes
//     assert_eq!(y.shape(), x.shape(), "Tensors y and x must have the same shape");
//     assert_eq!(w.shape()[0], x.shape()[0], "The size of w must match the first dimension of x");
//     let n = x.size();
//     // Compute the RMS of x: sqrt(sum(x_i^2) / n)
//     let sum_of_squares: f32 = x.data().iter().map(|&x_i| x_i * x_i).sum();
//     let rms = (sum_of_squares / n as f32).sqrt() + epsilon;
//     // Normalize and apply to y: y = (x / rms) * w
//     for (i, xi) in x.data().iter().enumerate() {
//         let weight = w.data()[i % w.size()]; // Ensure the weight corresponds to the current element of x
//         unsafe {
//             y.data_mut()[i] = (xi / rms) * weight;
//         }
//     }
// }
// pub fn matmul(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
//     // Ensure the shapes of A, B, and C are compatible for matrix multiplication
//     assert_eq!(a.shape()[0], c.shape()[0], "Rows of A must match rows of C");
//     assert_eq!(b.shape()[0], a.shape()[1], "Columns of A must match rows of B");
//     assert_eq!(b.shape()[1], c.shape()[1], "Columns of B must match columns of C");

//     let m = a.shape()[0]; // Number of rows in A and C
//     let k = a.shape()[1]; // Number of columns in A and rows in B
//     let n = b.shape()[1]; // Number of columns in B and C

//     // Scale C by beta
//     unsafe {
//         for i in 0..m {
//             for j in 0..n {
//                 // Accessing the data using unsafe block
//                 unsafe {
//                     c.data_mut()[i * n + j] *= beta;
//                 }
//             }
//         }
//     }

//     // Perform matrix multiplication and update C
//     unsafe {
//         for i in 0..m {
//             for j in 0..n {
//                 let mut sum = 0.0;
//                 for k_idx in 0..k {
//                     sum += a.data()[i * k + k_idx] * b.data()[k_idx * n + j];
//                 }
//                 // Accessing the data using unsafe block
//                 unsafe {

//                     c.data_mut()[i * n + j] += alpha * sum;
//                 }
//             }
//         }
//     }
// }
// fn mlp(
//     residual: &mut Tensor<f32>,
//     hidden_states: &mut Tensor<f32>,
//     gate: &mut Tensor<f32>,
//     up: &mut Tensor<f32>,
//     w_up: &Tensor<f32>,
//     w_down: &Tensor<f32>,
//     w_gate: &Tensor<f32>,
//     rms_w: &Tensor<f32>,
//     eps: f32,
// ) {
// //    // Normalize the residual (assuming this is a LayerNorm operation)
// //    let hidden = rms_norm(residual, rms_w, eps); // Apply RMS norm on residual
    
// //    // Gate computation (assuming gate_weight is w_gate)
// //    *gate = hidden.matmul(&w_gate.t()); // Perform matrix multiplication

// //    // Upward projection (assuming up_weight is w_up)
// //    *up = hidden.matmul(&w_up.t()); // Matrix multiplication for upward direction

// //    // SwiGLU activation: gate * sigmoid(gate) * up
// //    let act = gate * sigmoid(gate) * up; // SwiGLU activation

// //    // Output projection (assuming down_weight is w_down)
// //    let output = act.matmul(&w_down.t()); // Matrix multiplication for downward direction

// //    // Residual connection
// //    *residual = output + residual; // Add residual connection to output

//     }
// /*############################################################################################################################# */
// #[test]
// pub fn test_mlp() {
//     let seq_len = 4;
//     let d = 2;
//     let di = 3;
//     let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
//     let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
//     let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
//     let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
//     let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
//     let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
//     let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
//     let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
//     let eps = 1e-6;
//     mlp(
//         &mut residual,
//         &mut hidden_states,
//         &mut gate_buf,
//         &mut up_buf,
//         &w_up,
//         &w_down,
//         &w_gate,
//         &rms_w,
//         eps,
//     );

//     assert!(residual.close_to(
//         &Tensor::<f32>::new(
//             vec![
//                 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
//                 1.7290739
//             ],
//             &vec![seq_len, d]
//         ),
//         1e-3
//     ))
// }

// #[test]
// pub fn test_load_safetensors() {
//     use std::path::PathBuf;
//     use crate::tensor::float_eq;
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let model = Llama::from_safetensors(model_dir);
//     assert_eq!(model.vocab, 2048);
//     assert_eq!(model.n_layers, 2);
//     assert_eq!(model.n_q_h, 8);
//     assert_eq!(model.n_kv_h, 4);
//     assert_eq!(model.d, 128);
//     assert_eq!(model.dqkv, 16);
//     assert_eq!(model.di, 384);

//     assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
//     assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
//     assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
//     assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
//     assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
//     assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
//     assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
//     assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
//     assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
//     assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
//     assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
//     assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

// }
use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
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

            todo!("self_attention(...)");
            todo!("down_proj matmul and add residual");

            todo!("mlp(...)");
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
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        
        todo!("实现文本生成");
        
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
    todo!("Implement self_attention");
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
    OP::rms_norm(hidden_states, residual, rms_w, eps);

    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.0);
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.0);

    let gate_size = gate.size();
    let gate_data = unsafe { gate.data_mut() };
    let up_data = up.data();
    for i in 0..gate_size {
        let sigmoid_gate = 1.0 / (1.0 + (-gate_data[i]).exp());
        gate_data[i] = gate_data[i] * sigmoid_gate * up_data[i];
    }

    let mut output = Tensor::<f32>::default(hidden_states.shape()); // Temporary buffer for output
    OP::matmul_transb(&mut output, 0., gate, w_down, 1.0);

    let residual_size = residual.size();
    let residual_data = unsafe { residual.data_mut() };
    let output_data = output.data();
    for i in 0..residual_size {
        residual_data[i] += output_data[i];
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
pub fn show_safetensors(){
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let config = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let model_file = std::fs::read(model_dir.join("model.safetensors")).unwrap();
    let safetensor = SafeTensors::deserialize(&model_file).unwrap();
    let v=safetensor.tensors();
    let names=v.iter().map(|v| v.0.clone()).collect::<Vec<_>>();
    names.iter().for_each(|v| println!("{}", v));
    println!("{:?}",config);

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