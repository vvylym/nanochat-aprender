//! BPE tokenizer implementation for nanochat
//!
//! This crate provides:
//! - Byte Pair Encoding (BPE) training
//! - Token encoding and decoding
//! - Vocabulary management
//! - Special token handling
//!
//! # Example
//!
//! ```no_run
//! use nanochat_tokenizer::Tokenizer;
//!
//! // Train a tokenizer
//! let corpus = vec!["hello world", "hello rust"];
//! let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500).unwrap();
//!
//! // Encode text
//! let ids = tokenizer.encode("hello world").unwrap();
//!
//! // Decode back
//! let text = tokenizer.decode(&ids).unwrap();
//! ```

pub mod bpe;
pub mod vocab;
pub mod special_tokens;

pub use bpe::BPE;
pub use vocab::{Vocabulary, VocabularyError};
pub use special_tokens::SpecialTokens;

use anyhow::{Context, Result};
use std::path::Path;

/// Main tokenizer interface combining BPE, vocabulary, and special tokens
///
/// This is the primary API for tokenization operations.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    bpe: BPE,
}

impl Tokenizer {
    /// Train a new tokenizer from an iterator of text
    ///
    /// # Arguments
    /// * `text_iterator` - Iterator over training text
    /// * `vocab_size` - Target vocabulary size
    ///
    /// # Returns
    /// A trained tokenizer
    pub fn train_from_iterator<I, S>(text_iterator: I, vocab_size: usize) -> Result<Self>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let bpe = BPE::train_from_iterator(text_iterator, vocab_size)
            .context("Failed to train BPE tokenizer")?;
        Ok(Self { bpe })
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.bpe.encode(text)
    }

    /// Encode text with special tokens
    ///
    /// # Arguments
    /// * `text` - Text to encode
    /// * `prepend` - Optional special token to prepend (e.g., "<|bos|>")
    /// * `append` - Optional special token to append (e.g., "<|eos|>")
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn encode_with_special(
        &self,
        text: &str,
        prepend: Option<&str>,
        append: Option<&str>,
    ) -> Result<Vec<u32>> {
        let mut ids = self.encode(text)?;

        if let Some(prepend_token) = prepend {
            if let Ok(prepend_id) = self.special_token_id(prepend_token) {
                ids.insert(0, prepend_id);
            }
        }

        if let Some(append_token) = append {
            if let Ok(append_id) = self.special_token_id(append_token) {
                ids.push(append_id);
            }
        }

        Ok(ids)
    }

    /// Encode a batch of texts
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    /// Vector of token ID vectors
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    /// * `ids` - Slice of token IDs
    ///
    /// # Returns
    /// Decoded text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.bpe.decode(ids)
    }

    /// Decode a batch of token ID sequences
    ///
    /// # Arguments
    /// * `ids_batch` - Slice of token ID vectors
    ///
    /// # Returns
    /// Vector of decoded texts
    pub fn decode_batch(&self, ids_batch: &[Vec<u32>]) -> Result<Vec<String>> {
        ids_batch.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Get the ID for a special token
    ///
    /// # Arguments
    /// * `token` - Special token string (e.g., "<|bos|>")
    ///
    /// # Returns
    /// Token ID if the token is a special token
    pub fn special_token_id(&self, token: &str) -> Result<u32> {
        self.bpe
            .vocabulary()
            .token_to_id(token)
            .context("Special token not found in vocabulary")
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }

    /// Load tokenizer from a directory
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer directory
    ///
    /// # Returns
    /// Loaded tokenizer
    ///
    /// # Errors
    /// Returns an error if the tokenizer file cannot be read or parsed
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::fs;
        use serde_json;
        
        let path = path.as_ref();
        let tokenizer_file = path.join("tokenizer.json");
        
        if !tokenizer_file.exists() {
            anyhow::bail!("Tokenizer file not found: {}", tokenizer_file.display());
        }
        
        let _content = fs::read_to_string(&tokenizer_file)
            .with_context(|| format!("Failed to read tokenizer file: {}", tokenizer_file.display()))?;
        
        // For now, use JSON format. Later can switch to aprender .apr format
        let _data: serde_json::Value = serde_json::from_str(&_content)
            .context("Failed to parse tokenizer JSON")?;
        
        // TODO: Deserialize BPE from JSON
        // This is a placeholder - full implementation requires serializing/deserializing
        // vocabulary, merges, and special tokens
        anyhow::bail!("Tokenizer deserialization not yet fully implemented")
    }

    /// Save tokenizer to a directory
    ///
    /// # Arguments
    /// * `path` - Path to save directory
    ///
    /// # Errors
    /// Returns an error if the directory cannot be created or the tokenizer cannot be serialized
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use std::fs;
        use serde_json;
        
        let path = path.as_ref();
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory: {}", path.display()))?;
        
        let tokenizer_file = path.join("tokenizer.json");
        
        // Serialize tokenizer data
        let data = serde_json::json!({
            "vocab_size": self.vocab_size(),
            "merges": self.bpe.merges(),
            // TODO: Serialize full vocabulary and special tokens
        });
        
        let content = serde_json::to_string_pretty(&data)
            .context("Failed to serialize tokenizer")?;
        
        fs::write(&tokenizer_file, content)
            .with_context(|| format!("Failed to write tokenizer file: {}", tokenizer_file.display()))?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_train_and_encode() {
        let corpus = vec!["hello world", "hello rust"];
        let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500).unwrap();
        
        let ids = tokenizer.encode("hello").unwrap();
        assert!(!ids.is_empty());
    }
}
