//! GPT model implementation

use crate::config::GPTConfig;

/// GPT model
pub struct GPT {
    config: GPTConfig,
}

impl GPT {
    /// Create a new GPT model
    pub fn new(config: GPTConfig) -> Self {
        Self { config }
    }
}

