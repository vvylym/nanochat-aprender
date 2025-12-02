//! Core GPT model implementation for nanochat
//!
//! This crate provides the core transformer architecture including:
//! - Multi-head attention with Group-Query Attention (GQA)
//! - MLP with ReLU² activation
//! - Rotary Position Embeddings (RoPE)
//! - RMSNorm normalization
//! - Untied weights architecture
//!
//! # Example
//!
//! ```no_run
//! use nanochat_model::{GPTConfig, GPT, attention::KVCache};
//! use aprender::autograd::Tensor;
//!
//! // Create model configuration
//! let config = GPTConfig::default();
//!
//! // Create model
//! let mut model = GPT::new(config);
//!
//! // Create input token IDs [batch=1, seq_len=10]
//! let input_ids = Tensor::zeros(&[1, 10]);
//!
//! // Forward pass for inference (no targets)
//! let logits = model.forward_cache(&input_ids, None)?;
//! // logits shape: [1, 10, vocab_size]
//!
//! // Forward pass with KV cache for autoregressive generation
//! let mut kv_cache = KVCache::new();
//! let first_token = Tensor::zeros(&[1, 1]);
//! let logits1 = model.forward_cache(&first_token, Some(&mut kv_cache))?;
//!
//! // Next token (reusing cache)
//! let next_token = Tensor::zeros(&[1, 1]);
//! let logits2 = model.forward_cache(&next_token, Some(&mut kv_cache))?;
//!
//! // Save checkpoint
//! use nanochat_model::save_checkpoint;
//! save_checkpoint(&model, "checkpoint.json", None)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod attention;
pub mod checkpoint;
pub mod config;
pub mod gpt;
pub mod mlp;
pub mod norm;
pub mod rope;
pub mod stability;

// Public API exports

/// Model checkpoint management
///
/// Functions for saving and loading model checkpoints, including weights,
/// configuration, and training metadata.
pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointMetadata};

/// Model configuration
///
/// Defines the architecture hyperparameters for the GPT model, including
/// layer count, attention heads, embedding dimensions, and vocabulary size.
pub use config::GPTConfig;

/// GPT model implementation
///
/// The main transformer model with decoder-only architecture, supporting
/// both training and inference modes with KV cache for efficient generation.
pub use gpt::GPT;

/// Attention mechanism components
///
/// Provides multi-head attention with Group-Query Attention (GQA), QK normalization,
/// and KV cache for efficient autoregressive inference.
pub use attention::{apply_qk_norm, CausalSelfAttention, KVCache};

// Re-export common types for convenience
/// Error type alias for error handling
pub use anyhow::Error;
/// Result type alias for error handling
pub use anyhow::Result;

/// Validate that a tokenizer and model config are compatible
///
/// This function checks that the tokenizer's vocabulary size matches the model's
/// `vocab_size` configuration. Mismatched vocabulary sizes will cause runtime
/// errors when token IDs exceed the model's vocabulary size.
///
/// # Arguments
/// * `tokenizer` - The tokenizer to validate (must implement `vocab_size()`)
/// * `config` - The model configuration
///
/// # Errors
/// Returns an error if vocab_size doesn't match
///
/// # Example
///
/// ```no_run
/// use nanochat_model::{GPTConfig, validate_tokenizer_model_compatibility};
/// use aprender::text::tokenize::BpeTokenizer;
///
/// let tokenizer = BpeTokenizer::train(&["hello world"], 500)?;
/// let config = GPTConfig::with_tokenizer_vocab_size(
///     tokenizer.vocab_size(),
///     1024, 12, 6, 6, 768, None, None
/// )?;
///
/// // Validate compatibility
/// validate_tokenizer_model_compatibility(&tokenizer, &config)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn validate_tokenizer_model_compatibility(
    tokenizer: &aprender::text::tokenize::BpeTokenizer,
    config: &GPTConfig,
) -> Result<()> {
    let tokenizer_vocab = tokenizer.vocab_size();
    config
        .validate_vocab_size(tokenizer_vocab)
        .map_err(|e| anyhow::anyhow!("Tokenizer-model incompatibility: {}", e))?;
    Ok(())
}
