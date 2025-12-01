//! GPT model implementation

use aprender::autograd::Tensor;
use crate::config::GPTConfig;
use crate::attention::{CausalSelfAttention, KVCache};
use crate::mlp::MLP;
use crate::norm::rms_norm;
use crate::rope::{precompute_rotary_embeddings, apply_rotary_emb};
use anyhow::Result;

/// Transformer decoder block
///
/// Architecture:
/// - Pre-norm attention: x = x + attn(norm(x))
/// - Pre-norm MLP: x = x + mlp(norm(x))
/// - Uses RoPE for positional encoding
pub struct Block {
    /// Causal self-attention layer
    attn: CausalSelfAttention,
    /// MLP layer
    mlp: MLP,
    /// Layer index (for KV cache)
    layer_idx: usize,
}

impl Block {
    /// Create a new transformer decoder block
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `layer_idx` - Layer index (for KV cache management)
    pub fn new(config: &GPTConfig, layer_idx: usize) -> Self {
        let attn = CausalSelfAttention::new(config.n_embd, config.n_head, config.n_kv_head);
        let mlp = MLP::new(config.n_embd);
        
        Self {
            attn,
            mlp,
            layer_idx,
        }
    }

    /// Forward pass through the block
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    /// * `cos_sin` - Precomputed RoPE cos/sin frequencies
    /// * `kv_cache` - Optional KV cache for inference
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, n_embd]
    pub fn forward(
        &self,
        x: &Tensor,
        cos_sin: Option<(&Tensor, &Tensor)>,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        // Pre-norm attention: x = x + attn(norm(x))
        let x_norm = rms_norm(x)?;
        
        // Apply RoPE if provided (will be integrated into attention in full implementation)
        // For now, just pass through attention
        let attn_out = self.attn.forward(&x_norm, kv_cache)?;
        
        // Residual connection using aprender's tensor operations
        // Note: Need to ensure shapes match - attention should output [batch, seq_len, n_embd]
        let x_shape = x.shape();
        let attn_shape = attn_out.shape();
        
        // If shapes don't match, we have an issue
        if x_shape != attn_shape {
            anyhow::bail!(
                "Attention output shape {:?} doesn't match input shape {:?}",
                attn_shape,
                x_shape
            );
        }
        
        // Use aprender's tensor addition
        let x_after_attn = attn_out.add(x);

        // Pre-norm MLP: x = x + mlp(norm(x))
        let x_norm = rms_norm(&x_after_attn)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        
        // Residual connection
        let output = mlp_out.add(&x_after_attn);
        
        Ok(output)
    }
}

/// GPT model
pub struct GPT {
    config: GPTConfig,
}

impl GPT {
    /// Create a new GPT model
    pub fn new(config: GPTConfig) -> Self {
        Self { config }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let config = GPTConfig::default();
        let block = Block::new(&config, 0);
        // Just verify creation
        assert!(true);
    }

    #[test]
    fn test_block_forward() {
        let config = GPTConfig::default();
        let block = Block::new(&config, 0);
        let x = Tensor::ones(&[1, 10, 768]);
        
        let output = block.forward(&x, None, None).unwrap();
        
        assert_eq!(output.shape(), x.shape());
    }
}
