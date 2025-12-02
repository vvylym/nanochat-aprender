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

pub mod config;
pub mod gpt;
pub mod attention;
pub mod mlp;
pub mod rope;
pub mod norm;
pub mod checkpoint;
pub mod stability;

// Public API exports

/// Model checkpoint management
///
/// Functions for saving and loading model checkpoints, including weights,
/// configuration, and training metadata.
pub use checkpoint::{save_checkpoint, load_checkpoint, CheckpointMetadata};

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
pub use attention::{CausalSelfAttention, apply_qk_norm, KVCache};

// Re-export common types for convenience
/// Result type alias for error handling
pub use anyhow::Result;
/// Error type alias for error handling
pub use anyhow::Error;
