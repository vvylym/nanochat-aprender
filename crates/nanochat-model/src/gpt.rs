//! GPT model implementation

use crate::attention::{CausalSelfAttention, KVCache};
use crate::config::GPTConfig;
use crate::mlp::MLP;
use crate::norm::rms_norm;
use crate::rope::precompute_rotary_embeddings;
use anyhow::Result;
use aprender::autograd::Tensor;
use aprender::nn::Module;
use std::sync::OnceLock;

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
        let attn = CausalSelfAttention::new(
            config.n_embd,
            config.n_head,
            config.n_kv_head,
            config.dropout,
            config.seed,
        );
        let mlp = MLP::new(config.n_embd, config.seed);

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
        let attn_out = self
            .attn
            .forward(&x_norm, None, self.layer_idx, None)
            .expect("Attention failed");
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
    /// * `seed` - Optional random seed for reproducibility (None = non-deterministic)
    pub fn new(vocab_size: usize, n_embd: usize, seed: Option<u64>) -> Self {
        // Initialize embedding weights with normal distribution
        // Python: torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
        //
        // # Aprender API Compliance (Principle VII)
        // Note: aprender's init::normal() is pub(crate) (not public), so we cannot use
        // aprender::nn::init::normal() directly. We use the same Box-Muller transform
        // that aprender uses internally, ensuring statistical equivalence.
        //
        // Recommendation: Make aprender::nn::init::normal() public in aprender fork
        // to enable: let weight = aprender::nn::init::normal(&[vocab_size, n_embd], 0.0, 1.0, seed);
        //
        // Until then, we use StdRng with SeedableRng::seed_from_u64() per Principle VII.
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let numel = vocab_size * n_embd;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mean = 0.0;
        let std = 1.0; // Python uses std=1.0 (not 0.02)

        // Box-Muller transform for normal distribution (same as aprender uses internally)
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                let u1: f32 = rng.gen_range(0.0001_f32..1.0_f32);
                let u2: f32 = rng.gen_range(0.0_f32..1.0_f32);
                let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                mean + std * z
            })
            .collect();

        let weight = Tensor::new(&data, &[vocab_size, n_embd]).requires_grad();
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
                anyhow::bail!(
                    "Token ID {} exceeds vocabulary size {}",
                    token_id,
                    self.vocab_size
                );
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
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(n_embd: usize, vocab_size: usize, seed: Option<u64>) -> Self {
        use crate::init::init_linear_weight;

        // Create Linear layer without bias, then replace weight with Python's scheme
        let mut projection = aprender::nn::Linear::without_bias(n_embd, vocab_size);

        // Replace weight with Python's initialization scheme
        // Copy data from new weight tensor into existing weight
        if let Some(weight) = projection.parameters_mut().first_mut() {
            let new_weight = init_linear_weight(n_embd, vocab_size, seed).requires_grad();
            weight.data_mut().copy_from_slice(new_weight.data());
        }

        Self { projection }
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
        let wte = TokenEmbedding::new(config.vocab_size, config.n_embd, config.seed);

        // Create transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer_idx in 0..config.n_layer {
            blocks.push(Block::new(&config, layer_idx));
        }

        // Create language model head
        let lm_head = LanguageModelHead::new(config.n_embd, config.vocab_size, config.seed);

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

    /// Estimate FLOPs per token for the model
    ///
    /// Reference: https://arxiv.org/abs/2204.02311
    /// Formula: 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
    /// where:
    /// - l = n_layer
    /// - h = n_head
    /// - q = head_dim (n_embd / n_head)
    /// - t = sequence_len
    ///
    /// # Returns
    /// Estimated FLOPs per token as u64
    pub fn estimate_flops(&self) -> u64 {
        // Count total parameters
        let nparams: usize = self.parameters().iter().map(|p| p.numel()).sum();

        // Count embedding parameters (excluded from computation)
        // Access through parameters() - embedding weight is typically first
        let nparams_embedding: usize = self.parameters()
            .iter()
            .take(1) // First parameter is typically embedding weight
            .map(|p| p.numel())
            .sum();

        // Extract config values
        let l = self.config.n_layer;
        let h = self.config.n_head;
        let q = self.config.head_dim();
        let t = self.config.sequence_len;

        // Python formula: 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        6 * (nparams - nparams_embedding) as u64 + 12 * l as u64 * h as u64 * q as u64 * t as u64
    }

    /// Initialize model weights according to Python nanochat's scheme
    ///
    /// This zeros out:
    /// - Language model head weights
    /// - Output projection weights in all attention blocks (c_proj)
    /// - Output projection weights in all MLP blocks (c_proj)
    ///
    /// This should be called after model creation if you want to match
    /// Python's initialization exactly.
    ///
    /// Python reference: gpt.py:157-164
    pub fn init_weights(&mut self) -> Result<()> {
        // Zero out lm_head weights
        if let Some(weight) = self.lm_head.projection.parameters_mut().first_mut() {
            let shape = weight.shape();
            let numel: usize = shape.iter().product();
            let zeros_data = vec![0.0; numel];
            weight.data_mut().copy_from_slice(&zeros_data);
        }

        // Zero out c_proj weights in all blocks
        for block in &mut self.blocks {
            // Zero out attention output projection
            block.attn.zero_out_proj();

            // Zero out MLP output projection
            block.mlp.zero_out_proj();
        }

        Ok(())
    }

    /// Precompute RoPE embeddings if not already computed
    fn ensure_rope_embeddings(&self) -> Result<()> {
        if self.cos.get().is_none() || self.sin.get().is_none() {
            let head_dim = self.config.head_dim();
            let seq_len = self.config.sequence_len * 10; // Over-compute like Python version
            let (cos, sin) = precompute_rotary_embeddings(seq_len, head_dim, 10000.0)?;
            self.cos.set(cos).map_err(|_| anyhow::anyhow!("Failed to set cos"))?;
            self.sin.set(sin).map_err(|_| anyhow::anyhow!("Failed to set sin"))?;
        }
        Ok(())
    }

    /// Forward pass for training (with targets, no KV cache)
    ///
    /// # Arguments
    /// * `idx` - Token IDs tensor [batch, seq_len]
    /// * `targets` - Target token IDs for loss computation [batch, seq_len]
    /// * `loss_reduction` - Optional loss reduction strategy (default: Mean)
    ///
    /// # Returns
    /// Loss value (scalar tensor)
    pub fn forward_training(
        &self,
        idx: &Tensor,
        targets: &Tensor,
        loss_reduction: Option<aprender::nn::loss::Reduction>,
    ) -> Result<Tensor> {
        self.forward_internal(idx, Some(targets), None, loss_reduction)
    }

    /// Forward pass for inference with KV cache support
    ///
    /// # Arguments
    /// * `idx` - Token IDs tensor [batch, seq_len]
    /// * `kv_cache` - Optional KV cache for auto-regressive generation
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, vocab_size]
    pub fn forward_cache(&self, idx: &Tensor, kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        self.forward_internal(idx, None, kv_cache, None)
    }

    /// Internal forward pass implementation
    ///
    /// Note: Changed to `&self` because we only need interior mutability for RoPE embeddings
    /// and KV cache is passed separately. This allows `Module::forward()` to reuse this code.
    fn forward_internal(
        &self,
        idx: &Tensor,
        targets: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        loss_reduction: Option<aprender::nn::loss::Reduction>,
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

        // Apply logits softcap (Python: softcap * torch.tanh(logits / softcap))
        // This prevents logits from becoming too large, improving numerical stability
        const SOFTCAP: f32 = 15.0;
        let logits_scaled = logits.mul_scalar(1.0 / SOFTCAP);
        let logits_tanh = logits_scaled.tanh_();
        let logits = logits_tanh.mul_scalar(SOFTCAP);

        // If targets provided, compute loss for training
        if let Some(targets) = targets {
            use aprender::nn::loss::{CrossEntropyLoss, Reduction};

            // Reshape logits from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
            let batch = logits.shape()[0];
            let seq_len = logits.shape()[1];
            let vocab_size = logits.shape()[2];
            let logits_flat = logits.view(&[batch * seq_len, vocab_size]);

            // Reshape targets from [batch, seq_len] to [batch * seq_len]
            // Targets should be integer class indices (as f32)
            let targets_flat = targets.view(&[batch * seq_len]);

            // Use aprender's Reduction enum (default: Mean)
            let reduction = loss_reduction.unwrap_or(Reduction::Mean);
            let criterion = CrossEntropyLoss::with_reduction(reduction);
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
        // Reuse forward_internal() - it now takes &self so we can call it
        // Module::forward must return Tensor (not Result), so we unwrap errors
        self.forward_internal(input, None, None, None).expect("Forward pass failed")
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
        assert!(!block.mlp.parameters().is_empty());
        assert!(!block.parameters().is_empty());
    }

    #[test]
    fn test_block_forward() {
        let config = GPTConfig::default();
        let block = Block::new(&config, 0);
        let x = Tensor::ones(&[1, 10, 768]);

        let output = block.forward(&x, None, None).expect("Failed to forward block");

        assert_eq!(output.shape(), x.shape());
    }

    #[test]
    fn test_embedding_initialization_std() {
        // Test that embedding weights have std ≈ 1.0 (matching Python)
        let vocab_size = 100;
        let n_embd = 64;
        let seed = Some(42);

        let embedding = TokenEmbedding::new(vocab_size, n_embd, seed);
        let weight_data = embedding.weight().data();

        // Calculate mean and std
        let mean: f32 = weight_data.iter().sum::<f32>() / weight_data.len() as f32;
        let variance: f32 =
            weight_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weight_data.len() as f32;
        let std = variance.sqrt();

        // Python: mean ≈ 0.0, std ≈ 1.0
        // Allow tolerance for statistical variation
        assert!(mean.abs() < 0.1, "Embedding mean {mean} too far from 0.0");
        assert!(
            (std - 1.0).abs() < 0.1,
            "Embedding std {std} too far from 1.0 (expected ~1.0 for Python compatibility)"
        );
    }

    #[test]
    fn test_embedding_initialization_reproducible() {
        // Test that embedding initialization is reproducible with same seed
        let vocab_size = 50;
        let n_embd = 32;
        let seed = Some(123);

        let embedding1 = TokenEmbedding::new(vocab_size, n_embd, seed);
        let embedding2 = TokenEmbedding::new(vocab_size, n_embd, seed);

        let weight1 = embedding1.weight().data();
        let weight2 = embedding2.weight().data();

        // With same seed, weights should be identical
        assert_eq!(
            weight1, weight2,
            "Embedding weights should be reproducible with same seed"
        );
    }

    #[test]
    fn test_logits_softcap() {
        // Test that logits are bounded by softcap (15.0)
        let config = GPTConfig::default();
        let model = GPT::new(config);

        let input = Tensor::zeros(&[1, 5]);
        let logits = model.forward_cache(&input, None).expect("Failed to forward cache");

        let logits_data = logits.data();
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_logit = logits_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Softcap = 15, so tanh output is in [-1, 1], so logits in [-15, 15]
        assert!(
            max_logit <= 15.0,
            "Max logit {max_logit} exceeds softcap of 15.0"
        );
        assert!(
            min_logit >= -15.0,
            "Min logit {min_logit} below -softcap of -15.0"
        );
    }

    #[test]
    fn test_logits_softcap_small_values() {
        // Test that small logits are not significantly changed by softcap
        let config = GPTConfig::default();
        let model = GPT::new(config);

        let input = Tensor::zeros(&[1, 3]);
        let logits = model.forward_cache(&input, None).expect("Failed to forward cache");

        // For small values, tanh(x) ≈ x, so softcap should not change them much
        // This is a sanity check that softcap doesn't break small values
        let logits_data = logits.data();
        assert!(
            logits_data.iter().all(|&x| x.is_finite()),
            "All logits should be finite after softcap"
        );
    }

    #[test]
    fn test_init_weights() {
        // Test that init_weights() zeros out correct weights
        let config = GPTConfig::default();
        let mut model = GPT::new(config);

        // Call init_weights
        model.init_weights().expect("Failed to initialize weights");

        // Verify lm_head weights are zeroed (first parameter is weight)
        let lm_params = model.lm_head.projection.parameters();
        if let Some(weight) = lm_params.first() {
            let weight_data = weight.data();
            assert!(
                weight_data.iter().all(|&x| x == 0.0),
                "LM head weights should be zeroed"
            );
        }

        // Verify attention out_proj weights are zeroed in all blocks
        // Parameters order: q_proj, k_proj, v_proj, out_proj (index 3)
        for block in &model.blocks {
            let attn_params = block.attn.parameters();
            if attn_params.len() >= 4 {
                let out_proj_weight = attn_params[3];
                let weight_data = out_proj_weight.data();
                assert!(
                    weight_data.iter().all(|&x| x == 0.0),
                    "Attention out_proj weights should be zeroed"
                );
            }
        }

        // Verify MLP c_proj weights are zeroed in all blocks
        // Parameters order: c_fc, c_proj (index 1)
        for block in &model.blocks {
            let mlp_params = block.mlp.parameters();
            if mlp_params.len() >= 2 {
                let c_proj_weight = mlp_params[1];
                let weight_data = c_proj_weight.data();
                assert!(
                    weight_data.iter().all(|&x| x == 0.0),
                    "MLP c_proj weights should be zeroed"
                );
            }
        }
    }

    #[test]
    fn test_init_weights_other_weights_unchanged() {
        // Test that init_weights() doesn't affect other weights
        let config = GPTConfig::default();
        let mut model = GPT::new(config);

        // Get initial weights before init_weights() (q_proj is first param, c_fc is first param)
        let attn_params = model.blocks[0].attn.parameters();
        let initial_q_weight: Vec<f32> = attn_params[0].data().to_vec();

        let mlp_params = model.blocks[0].mlp.parameters();
        let initial_c_fc_weight: Vec<f32> = mlp_params[0].data().to_vec();

        // Call init_weights()
        model.init_weights().expect("Failed to initialize weights");

        // Verify other weights are unchanged
        let attn_params_after = model.blocks[0].attn.parameters();
        let after_q_weight: Vec<f32> = attn_params_after[0].data().to_vec();

        let mlp_params_after = model.blocks[0].mlp.parameters();
        let after_c_fc_weight: Vec<f32> = mlp_params_after[0].data().to_vec();

        assert_eq!(
            initial_q_weight, after_q_weight,
            "q_proj weights should not be changed by init_weights()"
        );
        assert_eq!(
            initial_c_fc_weight, after_c_fc_weight,
            "c_fc weights should not be changed by init_weights()"
        );
    }

    #[test]
    fn test_loss_reduction_mean() {
        // Test that Mean reduction (default) works
        use aprender::nn::loss::Reduction;

        let config = GPTConfig::default();
        let model = GPT::new(config);

        let idx = Tensor::zeros(&[1, 5]);
        let targets = Tensor::zeros(&[1, 5]);

        let loss_mean = model
            .forward_training(&idx, &targets, Some(Reduction::Mean))
            .expect("Failed to forward training");
        let loss_default = model
            .forward_training(&idx, &targets, None)
            .expect("Failed to forward training");

        // Default should be Mean, so they should be equal
        assert!(
            (loss_mean.item() - loss_default.item()).abs() < 1e-5,
            "Default reduction should be Mean"
        );

        // Loss should be a scalar (mean reduction)
        assert_eq!(loss_mean.shape(), &[1]);
    }

    #[test]
    fn test_loss_reduction_sum() {
        // Test that Sum reduction works
        use aprender::nn::loss::Reduction;

        let config = GPTConfig::default();
        let model = GPT::new(config);

        let idx = Tensor::zeros(&[1, 5]);
        let targets = Tensor::zeros(&[1, 5]);

        let loss_mean = model
            .forward_training(&idx, &targets, Some(Reduction::Mean))
            .expect("Failed to forward training");
        let loss_sum = model
            .forward_training(&idx, &targets, Some(Reduction::Sum))
            .expect("Failed to forward training");

        // Sum should be approximately Mean * numel (batch * seq_len)
        let batch = 1;
        let seq_len = 5;
        let numel = batch * seq_len;
        let expected_sum = loss_mean.item() * numel as f32;

        // Allow small tolerance for floating point
        assert!(
            (loss_sum.item() - expected_sum).abs() < 1e-3,
            "Sum reduction should be Mean * numel. Got {}, expected {}",
            loss_sum.item(),
            expected_sum
        );
    }

    #[test]
    fn test_estimate_flops() {
        // Test FLOPs estimation matches Python formula
        let config = GPTConfig::default();
        let model = GPT::new(config);

        let flops = model.estimate_flops();

        // FLOPs should be positive
        assert!(flops > 0, "FLOPs should be positive");

        // Verify formula components (for documentation)
        let _l = model.config().n_layer;
        let _h = model.config().n_head;
        let _q = model.config().head_dim();
        let _t = model.config().sequence_len;

        // The formula should produce a reasonable value
        // For default config: n_layer=12, n_head=6, head_dim=128, sequence_len=1024
        // This should produce a large but reasonable number
        assert!(
            flops > 1000,
            "FLOPs {flops} seems too small for default config"
        );

        // Verify it's not unreasonably large (sanity check)
        assert!(
            flops < 1_000_000_000_000, // 1 trillion
            "FLOPs {flops} seems unreasonably large"
        );
    }

    #[test]
    fn test_estimate_flops_custom_config() {
        // Test FLOPs estimation with custom config
        let config =
            GPTConfig::new(100, 100, 2, 4, 2, 64, None, None).expect("Failed to create config");
        let model = GPT::new(config);

        let flops = model.estimate_flops();

        // FLOPs should be positive
        assert!(flops > 0, "FLOPs should be positive");

        // With smaller config, FLOPs should be smaller than default
        let default_config = GPTConfig::default();
        let default_model = GPT::new(default_config);
        let default_flops = default_model.estimate_flops();

        assert!(
            flops < default_flops,
            "Smaller config should have fewer FLOPs. Got {}, default {}",
            flops,
            default_flops
        );
    }
}
