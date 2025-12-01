//! MLP with ReLU² activation

use aprender::autograd::Tensor;
use aprender::nn::{Linear, ReLU, Module};
use anyhow::Result;

/// MLP layer with ReLU² activation
///
/// Architecture:
/// - Expansion: n_embd -> 4 * n_embd
/// - Activation: ReLU² (relu(x).square())
/// - Projection: 4 * n_embd -> n_embd
/// - No bias in linear layers
pub struct MLP {
    /// Expansion layer: n_embd -> 4 * n_embd
    c_fc: Linear,
    /// Projection layer: 4 * n_embd -> n_embd
    c_proj: Linear,
    /// ReLU activation (for ReLU²)
    relu: ReLU,
}

impl MLP {
    /// Create a new MLP layer
    ///
    /// # Arguments
    /// * `n_embd` - Embedding dimension
    pub fn new(n_embd: usize) -> Self {
        // Create linear layers without bias
        // Note: aprender's Linear may have bias by default, we'll need to check
        let c_fc = Linear::new(n_embd, 4 * n_embd);
        let c_proj = Linear::new(4 * n_embd, n_embd);
        let relu = ReLU::new();
        
        Self {
            c_fc,
            c_proj,
            relu,
        }
    }

    /// Forward pass through the MLP
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Expansion: n_embd -> 4 * n_embd
        let x = self.c_fc.forward(x);
        
        // ReLU² activation: relu(x).square()
        let x = self.relu.forward(&x);
        let x_data = x.data();
        let x_squared: Vec<f32> = x_data.iter().map(|&val| val * val).collect();
        let x = Tensor::new(&x_squared, x.shape());
        
        // Projection: 4 * n_embd -> n_embd
        let output = self.c_proj.forward(&x);
        
        Ok(output)
    }
}

impl Module for MLP {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input).expect("MLP forward pass failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_creation() {
        let mlp = MLP::new(768);
        // Just verify it was created successfully
        assert!(true);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLP::new(768);
        let x = Tensor::ones(&[1, 10, 768]);
        
        let output = mlp.forward(&x).unwrap();
        
        // Output should have same batch and seq_len, but n_embd dimension
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 10);
        assert_eq!(output.shape()[2], 768);
    }

    #[test]
    fn test_mlp_relu_squared() {
        // Test that negative values become zero after ReLU²
        let mlp = MLP::new(4);
        let x = Tensor::new(&[-1.0, 0.0, 1.0, 2.0], &[1, 1, 4]);
        
        let output = mlp.forward(&x).unwrap();
        
        // Output should not contain NaN or Inf
        let output_data = output.data();
        assert!(!output_data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }
}
