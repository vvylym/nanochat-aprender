//! RMSNorm normalization

use anyhow::Result;
use aprender::autograd::Tensor;
use aprender::nn::{Module, RMSNorm};

/// Apply RMSNorm normalization (purely functional, no learnable parameters)
///
/// RMSNorm: x / sqrt(mean(x^2) + eps)
///
/// This is a functional implementation that normalizes over the last dimension.
/// Unlike LayerNorm, RMSNorm does not center the mean (no subtraction of mean).
///
/// # Arguments
/// * `x` - Input tensor of shape [..., hidden_dim]
///
/// # Returns
/// Normalized tensor with same shape as input
pub fn rms_norm(x: &Tensor) -> Result<Tensor> {
    // Create a temporary RMSNorm without learnable parameters
    // This uses aprender's RMSNorm::without_affine which sets weight to 1.0
    let shape = x.shape();
    if shape.is_empty() {
        anyhow::bail!("Input tensor must have at least one dimension");
    }

    let hidden_dim = shape[shape.len() - 1];
    let norm = RMSNorm::without_affine(&[hidden_dim]);

    Ok(norm.forward(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_basic() {
        // Create a simple 2D tensor: [batch_size=2, hidden_dim=4]
        let x = Tensor::ones(&[2, 4]);
        let result = rms_norm(&x).expect("RMSNorm failed");

        // Result should have same shape
        assert_eq!(result.shape(), x.shape());
    }

    #[test]
    fn test_rms_norm_zero_input() {
        // Test with zero input - should handle gracefully
        let x = Tensor::zeros(&[2, 4]);
        let result = rms_norm(&x);
        // Should not panic, but may produce NaN/Inf which is acceptable for zero input
        assert!(result.is_ok());
    }
}
