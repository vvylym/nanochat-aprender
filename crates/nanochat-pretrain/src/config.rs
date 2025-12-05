//! Training configuration structures for pretraining
//!
//! This module provides configuration structures for loading training hyperparameters
//! from JSON files, replacing hardcoded values with configurable options.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete training configuration loaded from file
///
/// This structure matches the JSON config file format and contains all
/// hyperparameters needed for pretraining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigFile {
    /// Model configuration
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingHyperparams,
    /// Optimizer configuration
    pub optimizer: OptimizerHyperparams,
}

/// Model architecture configuration
///
/// Defines the GPT model architecture parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size (must match tokenizer)
    pub vocab_size: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Number of key-value heads (GQA - Group Query Attention)
    pub n_kv_head: usize,
    /// Model embedding dimension
    pub n_embd: usize,
    /// Maximum sequence length
    pub sequence_len: usize,
}

/// Training hyperparameters
///
/// Defines training loop parameters including batch size, gradient accumulation,
/// validation intervals, and gradient clipping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparams {
    /// Batch size per device
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum training steps
    pub max_steps: usize,
    /// Checkpoint save interval (steps)
    pub save_interval: usize,
    /// Logging interval (steps)
    pub log_interval: usize,
    /// Validation evaluation interval (steps, 0 = disabled)
    pub eval_interval: usize,
    /// Number of tokens for validation evaluation
    pub eval_tokens: usize,
    /// Gradient clipping threshold (0.0 = disabled)
    pub grad_clip: f32,
}

/// Optimizer hyperparameters
///
/// Defines optimizer and learning rate scheduler parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerHyperparams {
    /// Learning rate
    pub learning_rate: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// AdamW beta1
    pub beta1: f32,
    /// AdamW beta2
    pub beta2: f32,
    /// AdamW epsilon
    pub eps: f32,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Maximum steps (for scheduler)
    pub max_steps: usize,
    /// Minimum learning rate (for cosine decay)
    pub min_lr: f32,
    /// Warmup ratio (alternative to warmup_steps, calculated as ratio of max_steps)
    pub warmup_ratio: Option<f32>,
    /// Warmdown ratio (for cosine decay, calculated as ratio of max_steps)
    pub warmdown_ratio: Option<f32>,
    /// Final LR fraction (for cosine decay, final LR = initial LR * final_lr_frac)
    pub final_lr_frac: Option<f32>,
}

impl TrainingConfigFile {
    /// Load configuration from JSON file
    ///
    /// # Arguments
    /// * `path` - Path to JSON configuration file
    ///
    /// # Returns
    /// Loaded configuration or error if file cannot be read or parsed
    ///
    /// # Example
    /// ```no_run
    /// use nanochat_pretrain::config::TrainingConfigFile;
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = TrainingConfigFile::from_file(Path::new("config.json"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;
        let config: TrainingConfigFile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path))?;
        Ok(config)
    }

    /// Create default configuration
    ///
    /// Returns a configuration with sensible defaults matching the Python
    /// reference implementation's default values.
    ///
    /// # Returns
    /// Default configuration instance
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            model: ModelConfig {
                vocab_size: 50304,
                n_layer: 12,
                n_head: 6,
                n_kv_head: 6,
                n_embd: 768,
                sequence_len: 1024,
            },
            training: TrainingHyperparams {
                batch_size: 32,
                seq_len: 256,
                gradient_accumulation_steps: 1,
                max_steps: 10000,
                save_interval: 1000,
                log_interval: 100,
                eval_interval: 250,
                eval_tokens: 20 * 524288,
                grad_clip: 1.0,
            },
            optimizer: OptimizerHyperparams {
                learning_rate: 1e-4,
                weight_decay: 0.1,
                beta1: 0.9,
                beta2: 0.95,
                eps: 1e-8,
                warmup_steps: 1000,
                max_steps: 10000,
                min_lr: 1e-6,
                warmup_ratio: None,
                warmdown_ratio: Some(0.2),
                final_lr_frac: Some(0.0),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = TrainingConfigFile::default();
        assert_eq!(config.model.vocab_size, 50304);
        assert_eq!(config.training.batch_size, 32);
        assert_eq!(config.optimizer.learning_rate, 1e-4);
    }

    #[test]
    fn test_config_from_file() {
        let config_json = r#"{
            "model": {
                "vocab_size": 1000,
                "n_layer": 4,
                "n_head": 2,
                "n_kv_head": 2,
                "n_embd": 128,
                "sequence_len": 512
            },
            "training": {
                "batch_size": 16,
                "seq_len": 128,
                "gradient_accumulation_steps": 2,
                "max_steps": 5000,
                "save_interval": 500,
                "log_interval": 50,
                "eval_interval": 100,
                "eval_tokens": 1000000,
                "grad_clip": 0.5
            },
            "optimizer": {
                "learning_rate": 0.0002,
                "weight_decay": 0.05,
                "beta1": 0.85,
                "beta2": 0.99,
                "eps": 1e-7,
                "warmup_steps": 500,
                "max_steps": 5000,
                "min_lr": 1e-7,
                "warmup_ratio": null,
                "warmdown_ratio": 0.1,
                "final_lr_frac": 0.1
            }
        }"#;

        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(config_json.as_bytes()).expect("Failed to write config");
        file.flush().expect("Failed to flush");

        let config = TrainingConfigFile::from_file(file.path()).expect("Failed to load config");

        assert_eq!(config.model.vocab_size, 1000);
        assert_eq!(config.model.n_layer, 4);
        assert_eq!(config.training.batch_size, 16);
        assert_eq!(config.training.grad_clip, 0.5);
        assert_eq!(config.optimizer.learning_rate, 0.0002);
    }
}
