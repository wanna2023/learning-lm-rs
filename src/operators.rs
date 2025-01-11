use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}




// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
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
pub fn masked_softmax(y: &mut Tensor<f32>) {
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

// 第二题
// pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
//     todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
// }
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let (batch_size, vec_size) = x.shape(); // 获取 batch 和每个向量的大小

    // 遍历每个 batch
    for i in 0..batch_size {
        // 计算当前向量的平方和
        let mut sum_of_squares = 0.0;
        for j in 0..vec_size {
            let xi = x.get(i, j); // 获取 x[i, j] 的值
            sum_of_squares += xi * xi; // 累加平方和
        }
        
        // 计算均方根（RMS），并加上 epsilon 防止除零
        let rms = (sum_of_squares / vec_size as f32 + epsilon).sqrt();
        
        // 归一化并乘以权重 w_i
        for j in 0..vec_size {
            let xi = x.get(i, j); // 获取 x[i, j] 的值
            let wi = w.get(j);    // 获取 w[j] 的值
            // 归一化并存储到 y 中
            y.set(i, j, xi * wi / rms); // 将归一化后的结果存入 y
        }
    }
}




// // 第一题
// // y = silu(x) * y
// // hint: this is an element-wise operation
// pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
//     // let len = y.size();
//     // assert!(len == x.size());

//     // let _y = unsafe { y.data_mut() };
//     // let _x = x.data();

//     todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
// }
// 这里的 Tensor 是你代码中的张量类型，假设它有 data_mut() 和 data() 方法来获取可变和不可变的原始数据
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // 获取 x 和 y 的数据
    let x_data = x.data();
    let y_data = y.data_mut();
    // 遍历每个元素，计算 Silu 和最终的 SwiGLU
    let len = x.size();
    for i in 0..len {
        let xi = x_data[i];
        // 计算 Silu(xi)
        let silu_xi = xi / (1.0 + (-xi).exp());
        // 计算 SwiGLU
        y_data[i] = xi * silu_xi;
    }
}


// // 第三题
// // C = beta * C + alpha * A @ B^T
// // hint: You don't need to do an explicit transpose of B
// pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
//     todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
// }
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 确保输入矩阵的维度匹配
    assert_eq!(a.shape()[1], b.shape()[0], "Dimensions of A and B do not match for multiplication.");
    assert_eq!(c.shape()[0], a.shape()[0], "Dimensions of C and A do not match.");
    assert_eq!(c.shape()[1], b.shape()[0], "Dimensions of C and B^T do not match.");

    // 第一步：C = beta * C
    // 如果beta不为零，就对现有的C做缩放
    for i in 0..c.shape()[0] {
        for j in 0..c.shape()[1] {
            c[(i, j)] *= beta;
        }
    }

    // 第二步：计算 A @ B^T 并加上 alpha * A @ B^T
    // 这里 B 的转置是通过矩阵乘法的规则隐式实现的
    for i in 0..a.shape()[0] {
        for j in 0..b.shape()[0] {
            let mut sum = 0.0f32;
            for k in 0..a.shape()[1] {
                sum += a[(i, k)] * b[(j, k)];
            }
            c[(i, j)] += alpha * sum;
        }
    }
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
