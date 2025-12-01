//! Model configuration

/// GPT model configuration
#[derive(Debug, Clone)]
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
        }
    }
}

