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
//! use nanochat_model::GPTConfig;
//! use nanochat_model::GPT;
//!
//! // Create model configuration
//! let config = GPTConfig::default();
//!
//! // Create model
//! let model = GPT::new(config);
//!
//! // Forward pass
//! // let output = model.forward(&input_ids, &kv_cache)?;
//! ```

pub mod config;
pub mod gpt;
pub mod attention;
pub mod mlp;
pub mod rope;
pub mod norm;
pub mod checkpoint;

// Public API exports
pub use config::GPTConfig;
pub use gpt::GPT;
pub use attention::{CausalSelfAttention, apply_qk_norm, KVCache};

// Re-export common types for convenience
pub use anyhow::Result;
pub use anyhow::Error;
