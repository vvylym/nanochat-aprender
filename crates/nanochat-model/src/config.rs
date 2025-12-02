//! Model configuration

use thiserror::Error;

/// Errors that can occur during configuration validation
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("n_embd ({n_embd}) must be divisible by n_head ({n_head})")]
    InvalidHeadDimension { n_embd: usize, n_head: usize },
    #[error("n_kv_head ({n_kv_head}) must be less than or equal to n_head ({n_head})")]
    InvalidKvHeadCount { n_kv_head: usize, n_head: usize },
    #[error("n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head}) for GQA")]
    InvalidGqaRatio { n_head: usize, n_kv_head: usize },
    #[error("vocab_size must be greater than 0, got {0}")]
    InvalidVocabSize(usize),
    #[error("n_layer must be greater than 0, got {0}")]
    InvalidLayerCount(usize),
    #[error("sequence_len must be greater than 0, got {0}")]
    InvalidSequenceLength(usize),
    #[error("vocab_size mismatch: config has {config}, but tokenizer has {tokenizer}")]
    VocabSizeMismatch { config: usize, tokenizer: usize },
}

/// GPT model configuration
///
/// # Vocabulary Size Synchronization
///
/// The `vocab_size` field **must** match the vocabulary size of the tokenizer
/// used with this model. Mismatched vocabulary sizes will cause runtime errors
/// when token IDs exceed the model's vocabulary size.
///
/// Use `validate_vocab_size()` or `with_tokenizer_vocab_size()` to ensure
/// compatibility.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GPTConfig {
    /// Maximum sequence length
    pub sequence_len: usize,
    /// Vocabulary size - **must match tokenizer's vocab_size**
    pub vocab_size: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of query heads
    pub n_head: usize,
    /// Number of key/value heads (for GQA)
    pub n_kv_head: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Dropout probability (0.0 = no dropout)
    pub dropout: Option<f32>,
    /// Random seed for reproducibility (None = non-deterministic)
    pub seed: Option<u64>,
}

impl GPTConfig {
    /// Create a new GPTConfig with validation
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence_len: usize,
        vocab_size: usize,
        n_layer: usize,
        n_head: usize,
        n_kv_head: usize,
        n_embd: usize,
        dropout: Option<f32>,
        seed: Option<u64>,
    ) -> Result<Self, ConfigError> {
        let config = Self {
            sequence_len,
            vocab_size,
            n_layer,
            n_head,
            n_kv_head,
            n_embd,
            dropout,
            seed,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    ///
    /// # Errors
    /// Returns an error if validation fails
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0 {
            return Err(ConfigError::InvalidVocabSize(self.vocab_size));
        }
        if self.n_layer == 0 {
            return Err(ConfigError::InvalidLayerCount(self.n_layer));
        }
        if self.sequence_len == 0 {
            return Err(ConfigError::InvalidSequenceLength(self.sequence_len));
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(ConfigError::InvalidHeadDimension {
                n_embd: self.n_embd,
                n_head: self.n_head,
            });
        }
        if self.n_kv_head > self.n_head {
            return Err(ConfigError::InvalidKvHeadCount {
                n_kv_head: self.n_kv_head,
                n_head: self.n_head,
            });
        }
        if !self.n_head.is_multiple_of(self.n_kv_head) {
            return Err(ConfigError::InvalidGqaRatio {
                n_head: self.n_head,
                n_kv_head: self.n_kv_head,
            });
        }
        Ok(())
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Validate that this config's vocab_size matches a tokenizer's vocabulary size
    ///
    /// # Arguments
    /// * `tokenizer_vocab_size` - Vocabulary size from tokenizer
    ///
    /// # Errors
    /// Returns an error if vocab_size doesn't match
    pub fn validate_vocab_size(&self, tokenizer_vocab_size: usize) -> Result<(), ConfigError> {
        if self.vocab_size != tokenizer_vocab_size {
            return Err(ConfigError::VocabSizeMismatch {
                config: self.vocab_size,
                tokenizer: tokenizer_vocab_size,
            });
        }
        Ok(())
    }

    /// Create config from tokenizer vocabulary size (ensures vocab_size matches)
    ///
    /// # Arguments
    /// * `tokenizer_vocab_size` - Vocabulary size from tokenizer
    /// * `sequence_len` - Maximum sequence length
    /// * `n_layer` - Number of transformer layers
    /// * `n_head` - Number of query heads
    /// * `n_kv_head` - Number of key/value heads (for GQA)
    /// * `n_embd` - Embedding dimension
    /// * `dropout` - Dropout probability (None = no dropout)
    /// * `seed` - Random seed for reproducibility (None = non-deterministic)
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid
    #[allow(clippy::too_many_arguments)]
    pub fn with_tokenizer_vocab_size(
        tokenizer_vocab_size: usize,
        sequence_len: usize,
        n_layer: usize,
        n_head: usize,
        n_kv_head: usize,
        n_embd: usize,
        dropout: Option<f32>,
        seed: Option<u64>,
    ) -> Result<Self, ConfigError> {
        Self::new(
            sequence_len,
            tokenizer_vocab_size, // Use tokenizer's vocab_size
            n_layer,
            n_head,
            n_kv_head,
            n_embd,
            dropout,
            seed,
        )
    }
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            sequence_len: 1024,
            vocab_size: 50304,
            n_layer: 12,
            n_head: 6,
            n_kv_head: 6,
            n_embd: 768,
            dropout: None,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_success() {
        let config = GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), None).unwrap();
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_config_validation_invalid_head_dim() {
        let result = GPTConfig::new(1024, 50304, 12, 5, 5, 768, Some(0.0), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_invalid_kv_head() {
        let result = GPTConfig::new(1024, 50304, 12, 6, 8, 768, Some(0.0), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_seed() {
        let config1 = GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), Some(42)).unwrap();
        let config2 = GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), Some(42)).unwrap();

        assert_eq!(config1.seed, Some(42));
        assert_eq!(config2.seed, Some(42));
        assert_eq!(config1.seed, config2.seed);
    }

    #[test]
    fn test_config_seed_none() {
        let config = GPTConfig::default();
        assert_eq!(config.seed, None);

        let config_with_seed =
            GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), Some(123)).unwrap();
        assert_eq!(config_with_seed.seed, Some(123));
    }
}
