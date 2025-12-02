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
}

/// GPT model configuration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GPTConfig {
    /// Maximum sequence length
    pub sequence_len: usize,
    /// Vocabulary size
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
    pub dropout: f32,
}

impl GPTConfig {
    /// Create a new GPTConfig with validation
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid
    pub fn new(
        sequence_len: usize,
        vocab_size: usize,
        n_layer: usize,
        n_head: usize,
        n_kv_head: usize,
        n_embd: usize,
        dropout: f32,
    ) -> Result<Self, ConfigError> {
        let config = Self {
            sequence_len,
            vocab_size,
            n_layer,
            n_head,
            n_kv_head,
            n_embd,
            dropout,
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
        if self.n_embd % self.n_head != 0 {
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
        if self.n_head % self.n_kv_head != 0 {
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
            dropout: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_success() {
        let config = GPTConfig::new(1024, 50304, 12, 6, 6, 768, 0.0).unwrap();
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_config_validation_invalid_head_dim() {
        let result = GPTConfig::new(1024, 50304, 12, 5, 5, 768, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_invalid_kv_head() {
        let result = GPTConfig::new(1024, 50304, 12, 6, 8, 768, 0.0);
        assert!(result.is_err());
    }
}

