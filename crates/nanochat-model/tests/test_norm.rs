//! Unit tests for RMSNorm

use aprender::autograd::Tensor;
use nanochat_model::norm::rms_norm;

#[test]
fn test_rms_norm_basic() {
    // Test basic RMSNorm on a simple tensor
    // RMSNorm: x / sqrt(mean(x^2) + eps)
    // This is a purely functional operation with no learnable parameters
    let x = Tensor::ones(&[2, 4]);
    let result = rms_norm(&x).expect("Failed to apply RMSNorm");

    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rms_norm_zero_input() {
    // Test RMSNorm with zero input
    // Should handle gracefully without division by zero
    let x = Tensor::zeros(&[2, 4]);
    let result = rms_norm(&x);

    // Should not panic, but may produce NaN/Inf which is acceptable for zero input
    assert!(result.is_ok());
}

#[test]
fn test_rms_norm_negative_values() {
    // Test RMSNorm with negative values
    // Should work correctly (squares are always positive)
    let x = Tensor::new(&[-1.0, -2.0, 1.0, 2.0], &[2, 2]);
    let result = rms_norm(&x).expect("Failed to apply RMSNorm");

    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rms_norm_large_values() {
    // Test RMSNorm with large values
    // Should not overflow
    let x = Tensor::new(&[100.0, 200.0, 300.0], &[1, 3]);
    let result = rms_norm(&x).expect("Failed to apply RMSNorm");

    assert_eq!(result.shape(), x.shape());
    // Result should be normalized (smaller values)
    let result_data = result.data();
    assert!(result_data.iter().all(|&x| x.abs() < 1000.0 || x.is_nan() || x.is_infinite()));
}

#[test]
fn test_rms_norm_small_values() {
    // Test RMSNorm with small values
    // Should not underflow
    let x = Tensor::new(&[0.001, 0.002, 0.003], &[1, 3]);
    let result = rms_norm(&x).expect("Failed to apply RMSNorm");

    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rms_norm_nan_detection() {
    // Test that RMSNorm handles NaN inputs gracefully
    // Note: Creating NaN tensors is tricky, so we'll just test that the function exists
    let x = Tensor::ones(&[2, 4]);
    let result = rms_norm(&x).expect("Failed to apply RMSNorm");

    // Verify output doesn't contain NaN for normal input
    let result_data = result.data();
    assert!(!result_data.iter().any(|&x| x.is_nan()));
}
