// use crate::tensor::Tensor;

// // get (row) vectors from a 2D table given a list of indices
// pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
//     let length = indices.size();
//     let table_shape = table.shape();
//     assert!(table_shape.len() == 2);
//     let dim = table_shape[1];
//     assert!(y.size() == length * dim);
//     for i in 0..length {
//         let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
//         let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
//         dst.copy_from_slice(src);
//     }
// }




// // RoPE: Rotary Positional Embedding
// pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
//     let shape = y.shape();
//     assert!(shape.len() == 3);
//     let seq_len = shape[0];
//     let n_heads = shape[1];
//     let d = shape[2];
//     let data = unsafe { y.data_mut() };
//     for tok in 0..seq_len {
//         let pos = start_pos + tok;
//         for head in 0..n_heads {
//             for i in 0..d / 2 {
//                 let a = data[tok * n_heads * d + head * d + i];
//                 let b = data[tok * n_heads * d + head * d + i + d / 2];
//                 let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
//                 let (sin, cos) = freq.sin_cos();
//                 data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
//                 data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
//             }
//         }
//     }
// }


// // softmax(x) = exp(x - max) / sum(exp(x - max))
// // y = softmax(mask(x))
// pub fn masked_softmax(y: &mut Tensor<f32>) {
//     let ndim = y.shape().len();
//     assert!(ndim >= 2);
//     let seq_len = y.shape()[ndim - 2];
//     let total_seq_len = y.shape()[ndim - 1];
//     let batch = y.size() / (seq_len * total_seq_len);
//     let data = unsafe { y.data_mut() };
//     for b in 0..batch {
//         let base = b * seq_len * total_seq_len;
//         for i in 0..seq_len {
//             let offset = base + i * total_seq_len;
//             let boundary = total_seq_len - seq_len + i + 1;

//             let max = data[offset..offset + boundary]
//                 .iter()
//                 .fold(data[offset], |a, b| a.max(*b));

//             let sum = (0..boundary)
//                 .map(|j| {
//                     let e = (data[offset + j] - max).exp();
//                     data[offset + j] = e;
//                     e
//                 })
//                 .sum::<f32>();

//             (0..boundary).for_each(|j| data[offset + j] /= sum);
//             (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
//         }
//     }
// }







// /*###################################################################################################################################### */
// // 第二题
// // pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
// //     todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
// // }
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
// /*###################################################################################################################################### */
// // 第一题
// // y = silu(x) * y
// // hint: this is an element-wise operation
// // pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
// //     // let len = y.size();
// //     // assert!(len == x.size());
// //     // let _y = unsafe { y.data_mut() };
// //     // let _x = x.data();
// //     todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
// // }
// pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
//     assert_eq!(y.shape(), x.shape(), "Tensors must have the same shape");
//     for i in 0..x.size() {
//         let xi = x.data()[i]; // x[i]
//         let si = xi * sigmoid(xi); // silu(xi) = xi * sigmoid(xi)
//         unsafe {
//             y.data_mut()[i] *= si; // y[i] = silu(xi) * y[i]
//         }
//     }
// }
// fn sigmoid(x: f32) -> f32 {
//     1.0 / (1.0 + (-x).exp())
// }

// //################################################################################################################################### */
// // 第三题
// // C = beta * C + alpha * A @ B^T
// // hint: You don't need to do an explicit transpose of B
// // pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
// //     todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
// // }
// pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
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

// //#####################################################################################################################################
// // Dot product of two tensors (treated as vectors)
// #[allow(unused)]
// pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
//     let len = x.size();
//     assert!(len == y.size());
//     let x_ = x.data();
//     let y_ = y.data();
//     let mut sum = 0.0;
//     for i in 0..len {
//         sum += x_[i] * y_[i];
//     }
//     sum
// }

// // Sample a index from a tensor (treated as a probability vector)
// pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
//     assert!(x.shape()[x.shape().len() - 1] == x.size());
//     if temperature <= 0. || top_k < 2 || top_p <= 0. {
//         return x
//             .data()
//             .iter()
//             .enumerate()
//             .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//             .unwrap()
//             .0 as _;
//     }

//     #[derive(Clone, Copy, PartialEq, Debug)]
//     struct Probability {
//         val: f32,
//         tok: u32,
//     }
//     impl Eq for Probability {}
//     impl PartialOrd for Probability {
//         #[inline]
//         fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//             Some(self.cmp(other))
//         }
//     }
//     impl Ord for Probability {
//         #[inline]
//         fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//             match self.val.total_cmp(&other.val) {
//                 std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
//                 ord => ord.reverse(),
//             }
//         }
//     }
//     impl From<(usize, &f32)> for Probability {
//         #[inline]
//         fn from((i, p): (usize, &f32)) -> Self {
//             Self {
//                 val: p.clone(),
//                 tok: i as _,
//             }
//         }
//     }

//     // sort
//     let mut logits = x
//         .data()
//         .iter()
//         .enumerate()
//         .map(Probability::from)
//         .collect::<Vec<_>>();
//     logits.sort_unstable();
//     let max = core::mem::replace(&mut logits[0].val, 1.);
//     // softmax & sum
//     for i in 1..logits.len() {
//         logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
//     }
//     // topk & topp & random
//     let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
//     let pp = logits[logits.len() - 1].val * top_p;
//     let plimit = rand::random::<f32>() * f32::min(pk, pp);
//     // sample
//     logits.iter().find(|p| p.val >= plimit).unwrap().tok
// }

// // Your implementation should at least pass the following tests:
// #[test]
// fn test_silu() {
//     let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
//     let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
//     swiglu(&mut y, &x);
//     assert!(y.close_to(
//         &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
//         1e-3
//     ));
// }

// #[test]
// fn test_rms_norm() {
//     let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
//     let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
//     let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
//     rms_norm(&mut y, &x, &w, 1e-6);
//     assert!(y.close_to(
//         &Tensor::<f32>::new(
//             vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
//             &vec![2, 2]
//         ),
//         1e-3
//     ));
// }

// #[test]
// fn test_matmul_transb() {
//     let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
//     let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
//     let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
//     matmul_transb(&mut c, 1., &a, &b, 1.);
//     assert!(c.close_to(
//         &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
//         1e-3
//     ));
// }
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

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let x_shape=x.shape();
    let y_shape=y.shape();

    let x_data=x.data();
    let w_data=w.data();
    let y_data=unsafe{y.data_mut()};

    let n = x_shape[x_shape.len() - 1];
    let num_vectors = x.size() / n;


    for i in 0..num_vectors {
        let offset = i * n;

        let sum_sq = x_data[offset..offset + n]
            .iter()
            .fold(0.0, |acc, &val| acc + val * val);
        let rms = (sum_sq / n as f32 + epsilon).sqrt();


        for j in 0..n {
            y_data[offset + j] = w_data[j] * (x_data[offset + j] / rms);
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
     let len = y.size();
     assert!(len == x.size());

     let y = unsafe { y.data_mut() };
     let x = x.data();
    
    for i in 0..len{
        let sigmoid_x=1.0/(1.0+(-x[i]).exp());
        let silu_x=sigmoid_x*x[i];
        y[i]*=silu_x;
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();

    assert_eq!(a_shape.len(), 2, "A must be a 2D matrix");
    assert_eq!(b_shape.len(), 2, "B must be a 2D matrix");
    assert_eq!(c_shape.len(), 2, "C must be a 2D matrix");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];


    assert_eq!(b_shape[1], k, "A's columns must match B's rows");
    assert!(c_shape[0] == m && c_shape[1] == n, "C's shape must be (m, n)");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    for i in 0..m {
        for j in 0..n {
            let mut dot_product = 0.0;
            for l in 0..k {
                dot_product += a_data[i * k + l] * b_data[j * k + l];
            }

            let index = i * n + j;
            c_data[index] = alpha * dot_product + beta * c_data[index];
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
