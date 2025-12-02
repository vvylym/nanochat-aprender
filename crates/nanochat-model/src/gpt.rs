//! GPT model implementation

use aprender::autograd::Tensor;
use aprender::nn::Module;
use crate::config::GPTConfig;
use crate::attention::{CausalSelfAttention, KVCache};
use crate::mlp::MLP;
use crate::norm::rms_norm;
// RoPE functions used in forward pass
use anyhow::Result;
use std::sync::OnceLock;

/// Initialize embedding weights with normal distribution
///
/// Uses N(0, 0.02) which is standard for transformer embeddings.
/// This provides better initialization than zeros.
fn init_embedding_weights(vocab_size: usize, n_embd: usize) -> Tensor {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let size = vocab_size * n_embd;
    let mut weights = Vec::with_capacity(size);
    let mut hasher = DefaultHasher::new();
    (vocab_size, n_embd).hash(&mut hasher);
    let mut seed = hasher.finish();
    
    // Box-Muller transform for normal distribution
    // N(0, 0.02) = 0.02 * N(0, 1)
    let std_dev = 0.02;
    let mut use_spare = false;
    let mut spare = 0.0;
    
    for _ in 0..size {
        if !use_spare {
            // Generate two uniform random numbers
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let u1 = ((seed >> 16) as f32 / 65536.0).max(1e-10).min(1.0 - 1e-10);
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let u2 = ((seed >> 16) as f32 / 65536.0).max(1e-10).min(1.0 - 1e-10);
            
            // Box-Muller transform
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin();
            
            weights.push(z0 * std_dev);
            spare = z1 * std_dev;
            use_spare = true;
        } else {
            weights.push(spare);
            use_spare = false;
        }
    }
    
    Tensor::new(&weights, &[vocab_size, n_embd])
}

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
        
        // Forward through attention with RoPE and KV cache
        let attn_out = self.attn.forward(&x_norm, kv_cache, self.layer_idx, cos_sin)?;
        
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

impl Module for Block {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Module::forward for compatibility, but our custom forward() handles KV cache
        // This is a simplified version without KV cache support
        // For full functionality, use Block::forward() directly
        let x_norm = rms_norm(input).expect("RMSNorm failed");
        let attn_out = self.attn.forward(&x_norm, None, self.layer_idx, None).expect("Attention failed");
        let x_after_attn = attn_out.add(input);
        let x_norm = rms_norm(&x_after_attn).expect("RMSNorm failed");
        let mlp_out = self.mlp.forward(&x_norm).expect("MLP failed");
        mlp_out.add(&x_after_attn)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        // Collect attention parameters
        params.extend(self.attn.parameters());
        // Collect MLP parameters
        params.extend(self.mlp.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        // Collect attention parameters
        params.extend(self.attn.parameters_mut());
        // Collect MLP parameters
        params.extend(self.mlp.parameters_mut());
        params
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
        // Initialize embedding weights with normal distribution
        // Standard practice: N(0, 0.02) for transformer embeddings
        // This provides better initialization than zeros
        let weight = init_embedding_weights(vocab_size, n_embd).requires_grad();
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

impl Module for TokenEmbedding {
    fn forward(&self, _input: &Tensor) -> Tensor {
        // Module::forward expects token_ids, but we need to handle the Result
        // For Module compatibility, we'll use the weight directly
        // Note: This is a simplified forward for Module trait compatibility
        // The actual embedding lookup is in the custom forward() method
        self.weight.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // TokenEmbedding has one parameter: the weight matrix
        // RMSNorm without_affine has no learnable parameters
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
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

impl Module for LanguageModelHead {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.projection.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.projection.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.projection.parameters_mut()
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
    /// Precomputed RoPE cos frequencies [1, max_seq_len, 1, head_dim/2]
    /// Using OnceLock for thread-safe lazy initialization
    cos: OnceLock<Tensor>,
    /// Precomputed RoPE sin frequencies [1, max_seq_len, 1, head_dim/2]
    /// Using OnceLock for thread-safe lazy initialization
    sin: OnceLock<Tensor>,
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
        // Using OnceLock for thread-safe lazy initialization
        let cos = OnceLock::new();
        let sin = OnceLock::new();

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

    /// Precompute RoPE embeddings if not already computed
    fn ensure_rope_embeddings(&mut self) -> Result<()> {
        if self.cos.get().is_none() || self.sin.get().is_none() {
            let head_dim = self.config.head_dim();
            let seq_len = self.config.sequence_len * 10; // Over-compute like Python version
            let (cos, sin) = crate::rope::precompute_rotary_embeddings(seq_len, head_dim, 10000.0)?;
            self.cos.set(cos).map_err(|_| anyhow::anyhow!("Failed to set cos"))?;
            self.sin.set(sin).map_err(|_| anyhow::anyhow!("Failed to set sin"))?;
        }
        Ok(())
    }

    /// Precompute RoPE embeddings if not already computed (for &self access)
    fn ensure_rope_embeddings_ref(&self) -> Result<()> {
        if self.cos.get().is_none() || self.sin.get().is_none() {
            let head_dim = self.config.head_dim();
            let seq_len = self.config.sequence_len * 10; // Over-compute like Python version
            let (cos, sin) = crate::rope::precompute_rotary_embeddings(seq_len, head_dim, 10000.0)?;
            self.cos.set(cos).map_err(|_| anyhow::anyhow!("Failed to set cos"))?;
            self.sin.set(sin).map_err(|_| anyhow::anyhow!("Failed to set sin"))?;
        }
        Ok(())
    }

    /// Forward pass through the GPT model
    ///
    /// # Arguments
    /// * `idx` - Token IDs tensor [batch, seq_len] with values in [0, vocab_size)
    /// * `targets` - Optional target token IDs for training [batch, seq_len]
    /// * `kv_cache` - Optional KV cache for inference
    ///
    /// # Returns
    /// - If `targets` is provided: Loss value (for training)
    /// - If `targets` is None: Logits tensor [batch, seq_len, vocab_size] (for inference)
    /// Forward pass for training (with targets, no KV cache)
    ///
    /// # Arguments
    /// * `idx` - Token IDs tensor [batch, seq_len]
    /// * `targets` - Target token IDs for loss computation [batch, seq_len]
    ///
    /// # Returns
    /// Loss value (scalar tensor)
    pub fn forward_training(
        &mut self,
        idx: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        self.forward_internal(idx, Some(targets), None)
    }
    
    /// Forward pass for inference with KV cache support
    ///
    /// # Arguments
    /// * `idx` - Token IDs tensor [batch, seq_len]
    /// * `kv_cache` - Optional KV cache for autoregressive generation
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, vocab_size]
    pub fn forward_cache(
        &mut self,
        idx: &Tensor,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        self.forward_internal(idx, None, kv_cache)
    }
    
    /// Internal forward pass implementation
    fn forward_internal(
        &mut self,
        idx: &Tensor,
        targets: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        // Ensure RoPE embeddings are computed
        self.ensure_rope_embeddings()?;

        let shape = idx.shape();
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor [batch, seq_len], got shape {:?}", shape);
        }

        let _batch = shape[0];
        let _seq_len = shape[1];

        // Get RoPE embeddings for current sequence length
        let cos = self.cos.get().expect("RoPE cos not initialized");
        let sin = self.sin.get().expect("RoPE sin not initialized"); 
        
        // RoPE slicing is now handled in CausalSelfAttention::forward()
        
        // Forward through token embedding with RMSNorm
        let x = self.wte.forward(idx)?;
        // x already has RMSNorm applied in wte.forward()
        
        // Forward through transformer blocks
        let mut x = x;
        
        // Handle KV cache by using indices instead of iterating directly
        // This allows us to pass mutable references properly
        if let Some(cache) = kv_cache {
            // Use unsafe to work around borrowing restrictions
            // This is safe because we're only accessing different layers sequentially
            // and each block only uses the cache for its specific layer
            let cache_ptr: *mut KVCache = cache;
            for layer_idx in 0..self.blocks.len() {
                let cos_sin = Some((cos, sin));
                let cache_ref = unsafe { &mut *cache_ptr };
                x = self.blocks[layer_idx].forward(&x, cos_sin, Some(cache_ref))?;
            }
        } else {
            for block in self.blocks.iter() {
                let cos_sin = Some((cos, sin));
                x = block.forward(&x, cos_sin, None)?;
            }
        }
        
        // Final RMSNorm
        let x = rms_norm(&x)?;
        
        // Forward through language model head to get logits
        let logits = self.lm_head.forward(&x)?;
        
        // If targets provided, compute loss for training
        if let Some(targets) = targets {
            use aprender::nn::loss::CrossEntropyLoss;
            
            // Reshape logits from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
            let batch = logits.shape()[0];
            let seq_len = logits.shape()[1];
            let vocab_size = logits.shape()[2];
            let logits_flat = logits.view(&[batch * seq_len, vocab_size]);
            
            // Reshape targets from [batch, seq_len] to [batch * seq_len]
            // Targets should be integer class indices (as f32)
            let targets_flat = targets.view(&[batch * seq_len]);
            
            // Compute cross-entropy loss
            let criterion = CrossEntropyLoss::new();
            let loss = criterion.forward(&logits_flat, &targets_flat);
            
            return Ok(loss);
        }
        
        Ok(logits)
    }
}

impl Module for GPT {
    /// Forward pass for Module trait compatibility
    ///
    /// This performs inference without KV cache for compatibility with
    /// aprender's Module trait. For full functionality, use:
    /// - `GPT::forward_training()` for training
    /// - `GPT::forward_cache()` for inference with KV cache
    fn forward(&self, input: &Tensor) -> Tensor {
        // Ensure RoPE embeddings are computed
        self.ensure_rope_embeddings_ref().expect("Failed to compute RoPE embeddings");

        let shape = input.shape();
        if shape.len() != 2 {
            panic!("Expected 2D tensor [batch, seq_len], got shape {:?}", shape);
        }

        // Get RoPE embeddings for current sequence length
        let cos_ref = self.cos.get().expect("RoPE cos not initialized");
        let sin_ref = self.sin.get().expect("RoPE sin not initialized");
        
        // Forward through token embedding with RMSNorm
        let x = self.wte.forward(input).expect("Token embedding failed");
        
        // Forward through transformer blocks (no KV cache for Module::forward)
        let mut x = x;
        for block in self.blocks.iter() {
            let cos_sin = Some((cos_ref, sin_ref));
            x = block.forward(&x, cos_sin, None).expect("Block forward failed");
        }
        
        // Final RMSNorm
        let x = rms_norm(&x).expect("RMSNorm failed");
        
        // Forward through language model head to get logits
        self.lm_head.forward(&x).expect("LM head forward failed")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        // Add embedding weights
        params.extend(self.wte.parameters());
        
        // Add block parameters
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        
        // Add LM head parameters
        params.extend(self.lm_head.parameters());
        
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        
        // Add embedding weights
        params.extend(self.wte.parameters_mut());
        
        // Add block parameters
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        
        // Add LM head parameters
        params.extend(self.lm_head.parameters_mut());
        
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let config = GPTConfig::default();
        let block = Block::new(&config, 0);
        
        // Verify block has attention and MLP
        assert_eq!(block.attn.n_head(), config.n_head);
        assert_eq!(block.attn.n_kv_head(), config.n_kv_head);
        assert!(block.mlp.parameters().len() > 0);
        assert!(block.parameters().len() > 0);
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
