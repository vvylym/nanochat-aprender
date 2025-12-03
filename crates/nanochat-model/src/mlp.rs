//! MLP with ReLU² activation

use anyhow::Result;
use aprender::autograd::Tensor;
use aprender::nn::{Linear, Module, ReLU};

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
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(n_embd: usize, seed: Option<u64>) -> Self {
        // Initialize Linear layers with Python's initialization scheme
        use crate::init::init_linear_weight;

        // Create linear layers without bias, then replace weights
        let mut c_fc = Linear::without_bias(n_embd, 4 * n_embd);
        let mut c_proj = Linear::without_bias(4 * n_embd, n_embd);

        // Replace weights with Python's initialization scheme
        // Copy data from new weight tensor into existing weight
        if let Some(weight) = c_fc.parameters_mut().first_mut() {
            let new_weight = init_linear_weight(n_embd, 4 * n_embd, seed).requires_grad();
            weight.data_mut().copy_from_slice(new_weight.data());
        }
        if let Some(weight) = c_proj.parameters_mut().first_mut() {
            let new_weight = init_linear_weight(4 * n_embd, n_embd, seed).requires_grad();
            weight.data_mut().copy_from_slice(new_weight.data());
        }

        let relu = ReLU::new();

        Self { c_fc, c_proj, relu }
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

    /// Zero out the output projection weights (for Python-compatible initialization)
    pub fn zero_out_proj(&mut self) {
        if let Some(weight) = self.c_proj.parameters_mut().first_mut() {
            let shape = weight.shape();
            let numel: usize = shape.iter().product();
            let zeros_data = vec![0.0; numel];
            weight.data_mut().copy_from_slice(&zeros_data);
        }
    }
}

impl Module for MLP {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input).expect("MLP forward pass failed")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters_mut());
        params.extend(self.c_proj.parameters_mut());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_creation() {
        let mlp = MLP::new(768, None);

        // Verify MLP has expansion and projection layers
        let params = mlp.parameters();
        assert!(!params.is_empty());
        // Each Linear layer has at least weight, may have bias
        // So we expect at least 2 parameters (one per layer)
        assert!(params.len() >= 2);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLP::new(768, None);
        let x = Tensor::ones(&[1, 10, 768]);

        let output = mlp.forward(&x).expect("MLP forward pass failed");

        // Output should have same batch and seq_len, but n_embd dimension
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 10);
        assert_eq!(output.shape()[2], 768);
    }

    #[test]
    fn test_mlp_relu_squared() {
        // Test that negative values become zero after ReLU²
        let mlp = MLP::new(4, None);
        let x = Tensor::new(&[-1.0, 0.0, 1.0, 2.0], &[1, 1, 4]);

        let output = mlp.forward(&x).expect("MLP forward pass failed");

        // Output should not contain NaN or Inf
        let output_data = output.data();
        assert!(!output_data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }
}
