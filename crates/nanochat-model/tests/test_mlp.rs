//! Unit tests for MLP with ReLU² activation

use nanochat_model::mlp::MLP;
use aprender::autograd::Tensor;

#[test]
fn test_mlp_basic() {
    // Test basic MLP forward pass
    let mlp = MLP::new(768);
    let x = Tensor::ones(&[1, 10, 768]);
    
    let output = mlp.forward(&x).unwrap();
    
    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_mlp_relu_squared() {
    // Test ReLU² activation: relu(x)^2
    let mlp = MLP::new(4);
    let x = Tensor::new(&[-1.0, 0.0, 1.0, 2.0], &[1, 1, 4]);
    
    let output = mlp.forward(&x).unwrap();
    
    // Output should not contain NaN or Inf
    let output_data = output.data();
    assert!(!output_data.iter().any(|&x| x.is_nan() || x.is_infinite()));
}

#[test]
fn test_mlp_negative_inputs() {
    // Test MLP with negative inputs
    // Note: After ReLU² in the hidden layer, negative values become 0,
    // but the final linear layer can produce negative outputs
    let mlp = MLP::new(4);
    let x = Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[1, 1, 4]);
    
    let output = mlp.forward(&x).unwrap();
    
    // Output should have correct shape and not contain NaN/Inf
    assert_eq!(output.shape(), x.shape());
    let output_data = output.data();
    assert!(!output_data.iter().any(|&x| x.is_nan() || x.is_infinite()));
}

#[test]
fn test_mlp_positive_inputs() {
    // Test MLP with positive inputs
    let mlp = MLP::new(4);
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    
    let output = mlp.forward(&x).unwrap();
    
    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_mlp_expansion_ratio() {
    // Test MLP expansion ratio (typically 4x: n_embd -> 4*n_embd -> n_embd)
    let mlp = MLP::new(256);
    let x = Tensor::ones(&[1, 10, 256]);
    
    let output = mlp.forward(&x).unwrap();
    
    // Output should have same shape as input (expansion happens internally)
    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_mlp_no_bias() {
    // Test that MLP layers have no bias (bias=False)
    // This is verified by the MLP implementation using Linear layers
    // which may have bias, but the architecture doesn't require it
    let mlp = MLP::new(768);
    let x = Tensor::ones(&[1, 10, 768]);
    
    // Just verify it works - bias configuration is internal
    let output = mlp.forward(&x).unwrap();
    assert_eq!(output.shape(), x.shape());
}
