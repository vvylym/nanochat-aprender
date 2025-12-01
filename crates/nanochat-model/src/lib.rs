//! Core GPT model implementation for nanochat
//!
//! This crate provides the core transformer architecture including:
//! - Multi-head attention with Group-Query Attention (GQA)
//! - MLP with ReLU² activation
//! - Rotary Position Embeddings (RoPE)
//! - RMSNorm normalization
//! - Untied weights architecture

pub mod config;
pub mod gpt;
pub mod attention;
pub mod mlp;
pub mod rope;
pub mod norm;
pub mod checkpoint;

pub use config::GPTConfig;
pub use gpt::GPT;

// Re-export for tests
#[cfg(test)]
pub use config::GPTConfig as ModelConfig;

