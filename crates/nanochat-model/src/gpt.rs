//! GPT model implementation

use aprender::autograd::Tensor;
use aprender::nn::Module;
use crate::config::GPTConfig;
use crate::attention::{CausalSelfAttention, KVCache};
use crate::mlp::MLP;
use crate::norm::rms_norm;
use crate::rope::{precompute_rotary_embeddings, apply_rotary_emb};
use anyhow::Result;

/// Tryseansformer decoder block
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

/// Token embedding layer with RMSNorm
///
/// This embeds token IDs into the embedding space and applies RMSNorm.
/// The embedding weights are untied from the language model head.
///
/// Note: Since aprender doesn't have an Embedding module, we implement
/// embedding as a lookup table (weight matrix) with indexing.
pub struct TokenEmbedding {
    /// Embedding weight matrix: [vocab_size, n_embd]
    /// This acts as a lookup table for token embeddings
    weight: Tensor,
    /// RMSNorm applied after embedding
    norm: aprender::nn::RMSNorm,
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    n_embd: usize,
}

impl TokenEmbedding {
    /// Create a new token embedding layer
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size
    /// * `n_embd` - Embedding dimension
    pub fn new(vocab_size: usize, n_embd: usize) -> Self {
        // Initialize embedding weights (will be learned during training)
        // For now, use zeros - will be initialized properly during model initialization
        let weight = Tensor::zeros(&[vocab_size, n_embd]);
        let norm = aprender::nn::RMSNorm::without_affine(&[n_embd]);
        
        Self {
            weight,
            norm,
            vocab_size,
            n_embd,
        }
    }

    /// Forward pass: embed tokens and apply RMSNorm
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs tensor [batch, seq_len] with values in [0, vocab_size)
    ///
    /// # Returns
    /// Embedded tokens [batch, seq_len, n_embd]
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let shape = token_ids.shape();
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor [batch, seq_len], got shape {:?}", shape);
        }
        
        let batch = shape[0];
        let seq_len = shape[1];
        
        // Lookup embeddings: for each token_id, get the corresponding row from weight matrix
        let token_data = token_ids.data();
        let weight_data = self.weight.data();
        let mut embedded_data = Vec::with_capacity(batch * seq_len * self.n_embd);
        
        for &token_id in token_data {
            let token_id = token_id as usize;
            if token_id >= self.vocab_size {
                anyhow::bail!("Token ID {} exceeds vocabulary size {}", token_id, self.vocab_size);
            }
            
            // Get embedding vector for this token
            let offset = token_id * self.n_embd;
            embedded_data.extend_from_slice(&weight_data[offset..offset + self.n_embd]);
        }
        
        // Reshape to [batch, seq_len, n_embd]
        let embedded = Tensor::new(&embedded_data, &[batch, seq_len, self.n_embd]);
        
        // Apply RMSNorm (using Module trait)
        let normalized = self.norm.forward(&embedded);
        
        Ok(normalized)
    }

    /// Get the embedding weight matrix (for parameter access)
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

/// Language model head (untied from embedding)
///
/// Projects from embedding space to vocabulary space for next-token prediction.
pub struct LanguageModelHead {
    /// Linear projection: n_embd -> vocab_size
    projection: aprender::nn::Linear,
}

impl LanguageModelHead {
    /// Create a new language model head
    ///
    /// # Arguments
    /// * `n_embd` - Embedding dimension
    /// * `vocab_size` - Vocabulary size
    pub fn new(n_embd: usize, vocab_size: usize) -> Self {
        let projection = aprender::nn::Linear::new(n_embd, vocab_size);
        
        Self {
            projection,
        }
    }

    /// Forward pass: project embeddings to vocabulary logits
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, vocab_size]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use aprender::nn::Module;
        Ok(self.projection.forward(x))
    }
}

/// GPT model with configurable depth
///
/// Architecture:
/// - Token embedding with RMSNorm
/// - N transformer decoder blocks
/// - Language model head (untied from embedding)
pub struct GPT {
    /// Model configuration
    config: GPTConfig,
    /// Token embedding layer
    wte: TokenEmbedding,
    /// Transformer decoder blocks
    blocks: Vec<Block>,
    /// Language model head
    lm_head: LanguageModelHead,
    /// Precomputed RoPE cos/sin frequencies
    cos: Option<Tensor>,
    sin: Option<Tensor>,
}

impl GPT {
    /// Create a new GPT model
    ///
    /// # Arguments
    /// * `config` - Model configuration
    pub fn new(config: GPTConfig) -> Self {
        // Validate configuration
        config.validate().expect("Invalid GPT configuration");

        // Create token embedding
        let wte = TokenEmbedding::new(config.vocab_size, config.n_embd);

        // Create transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer_idx in 0..config.n_layer {
            blocks.push(Block::new(&config, layer_idx));
        }

        // Create language model head
        let lm_head = LanguageModelHead::new(config.n_embd, config.vocab_size);

        // Precompute RoPE embeddings (will be computed on first forward pass)
        // For now, set to None - will be computed when needed
        let cos = None;
        let sin = None;

        Self {
            config,
            wte,
            blocks,
            lm_head,
            cos,
            sin,
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get the number of layers
    pub fn n_layer(&self) -> usize {
        self.config.n_layer
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
