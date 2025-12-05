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
//! let corpus = ["hello world", "hello rust"];
//! let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500).expect("Failed to train tokenizer");
//!
//! // Encode text
//! let ids = tokenizer.encode("hello world").expect("Encoding failed");
//!
//! // Decode back
//! let text = tokenizer.decode(&ids).expect("Decoding failed");
//! ```

// Re-export aprender types for backward compatibility
pub use aprender::text::tokenize::{BpeTokenizer, SpecialTokens};

use anyhow::{Context, Result};
use std::path::Path;

/// Tokenizer data
///
/// This struct is used to serialize and deserialize the tokenizer data.
/// It contains the vocabulary and merge rules for the tokenizer.
/// It is used to save and load the tokenizer data.
///
/// Match Python reference: only vocabulary and merges are needed
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TokenizerData {
    /// Token to ID mapping
    pub vocabulary: std::collections::HashMap<String, u32>,
    /// BPE merge rules
    pub merges: Vec<(String, String)>,
}

/// Main tokenizer interface combining BPE, vocabulary, and special tokens
///
/// This is the primary API for tokenization operations.
/// Uses `aprender::text::tokenize::BpeTokenizer` internally for BPE implementation.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    bpe: BpeTokenizer,
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
        // Convert iterator to owned strings first, then collect references
        // This ensures the strings live long enough for aprender's API
        let corpus_owned: Vec<String> = text_iterator.map(|s| s.as_ref().to_string()).collect();
        let corpus: Vec<&str> = corpus_owned.iter().map(|s| s.as_str()).collect();

        // Use aprender's BPE tokenizer
        let bpe = BpeTokenizer::train(&corpus, vocab_size)
            .map_err(|e| anyhow::anyhow!("Failed to train BPE tokenizer: {}", e))?;

        Ok(Self { bpe })
    }

    /// Train tokenizer from text files in a directory
    ///
    /// Reads all `.txt` files from the specified directory, loads their content,
    /// and trains a BPE tokenizer on the combined text.
    ///
    /// # Arguments
    /// * `data_dir` - Directory containing text files (.txt)
    /// * `vocab_size` - Target vocabulary size
    /// * `special_tokens` - Optional special tokens to add after training
    ///
    /// # Returns
    /// Trained tokenizer
    ///
    /// # Errors
    /// Returns an error if:
    /// - The directory cannot be read
    /// - No .txt files are found
    /// - Training fails
    ///
    /// # Example
    /// ```no_run
    /// use nanochat_tokenizer::Tokenizer;
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let special_tokens = vec![
    ///     "<|bos|>".to_string(),
    ///     "<|eos|>".to_string(),
    ///     "<|pad|>".to_string(),
    /// ];
    /// let tokenizer = Tokenizer::train_from_directory(
    ///     Path::new("./data"),
    ///     50000,
    ///     Some(special_tokens),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn train_from_directory(
        data_dir: &Path,
        vocab_size: usize,
        special_tokens: Option<Vec<String>>,
    ) -> Result<Self> {
        use std::fs;
        use std::io::Read;

        // Collect all text from .txt files
        let mut texts = Vec::new();
        let entries = fs::read_dir(data_dir)
            .with_context(|| format!("Failed to read directory: {:?}", data_dir))?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                let mut file = fs::File::open(&path)
                    .with_context(|| format!("Failed to open file: {:?}", path))?;
                let mut content = String::new();
                file.read_to_string(&mut content)
                    .with_context(|| format!("Failed to read file: {:?}", path))?;
                texts.push(content);
            }
        }

        if texts.is_empty() {
            anyhow::bail!("No .txt files found in directory: {:?}", data_dir);
        }

        // Create iterator over texts
        let text_iter = texts.iter().map(|s| s.as_str());

        // Train tokenizer using existing train_from_iterator method
        let tokenizer = Self::train_from_iterator(text_iter, vocab_size)
            .context("Failed to train tokenizer from data files")?;

        // Add special tokens if provided
        // Note: aprender's BpeTokenizer may handle special tokens during training
        // If not, we may need to add them manually. For now, we'll document this.
        if special_tokens.is_some() {
            // TODO: Check if aprender supports adding special tokens after training
            // If not, we may need to include them in the training corpus or use a different approach
            // For now, we'll log a warning if special tokens are provided
            eprintln!(
                "Warning: Special tokens provided but may not be added to trained tokenizer. \
                      Check aprender's BpeTokenizer API for special token support."
            );
        }

        Ok(tokenizer)
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.bpe.encode(text).map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))
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
        self.bpe.decode(ids).map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
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
            .token_to_id(token)
            .ok_or_else(|| anyhow::anyhow!("Special token not found: {}", token))
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
        use serde_json;
        use std::fs;

        let path = path.as_ref();
        let tokenizer_file = path.join("tokenizer.json");

        if !tokenizer_file.exists() {
            anyhow::bail!("Tokenizer file not found: {}", tokenizer_file.display());
        }

        let content = fs::read_to_string(&tokenizer_file).with_context(|| {
            format!(
                "Failed to read tokenizer file: {}",
                tokenizer_file.display()
            )
        })?;

        let data: TokenizerData =
            serde_json::from_str(&content).context("Failed to parse tokenizer JSON")?;

        // Reconstruct aprender's BpeTokenizer using from_vocab
        let vocab = data.vocabulary;
        let merges = data.merges;

        let bpe = BpeTokenizer::from_vocab(vocab, merges);

        Ok(Self { bpe })
    }

    /// Save tokenizer to a directory
    ///
    /// # Arguments
    /// * `path` - Path to save directory
    ///
    /// # Errors
    /// Returns an error if the directory cannot be created or the tokenizer cannot be serialized
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use serde_json;
        use std::fs;

        let path = path.as_ref();
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory: {}", path.display()))?;

        let tokenizer_file = path.join("tokenizer.json");

        // Serialize tokenizer data using TokenizerData struct for type safety and consistency
        // Match Python reference: only serialize vocabulary and merges
        // No version or vocab_size fields (vocab_size is derivable from vocabulary)
        let data = TokenizerData {
            vocabulary: self.bpe.vocab().clone(),
            merges: self.bpe.merges().to_vec(),
            // Special tokens are included in the vocabulary, no need to serialize separately
        };

        // Use compact JSON (not pretty-printed) to reduce file size
        let content = serde_json::to_string(&data).context("Failed to serialize tokenizer")?;

        fs::write(&tokenizer_file, content).with_context(|| {
            format!(
                "Failed to write tokenizer file: {}",
                tokenizer_file.display()
            )
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_train_and_encode() {
        let corpus = ["hello world", "hello rust"];
        let tokenizer =
            Tokenizer::train_from_iterator(corpus.iter(), 500).expect("Failed to train tokenizer");

        let ids = tokenizer.encode("hello").expect("Encoding failed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_train_from_directory() {
        use std::fs;
        use tempfile::TempDir;

        // Create temporary directory with test files
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file1 = temp_dir.path().join("data1.txt");
        let file2 = temp_dir.path().join("data2.txt");

        fs::write(&file1, "hello world hello rust").expect("Failed to write file1");
        fs::write(&file2, "world peace rust is awesome").expect("Failed to write file2");

        // Train tokenizer from directory
        let tokenizer = Tokenizer::train_from_directory(
            temp_dir.path(),
            500,
            Some(vec!["<|bos|>".to_string(), "<|eos|>".to_string()]),
        )
        .expect("Failed to train tokenizer from directory");

        // Verify tokenizer works
        let ids = tokenizer.encode("hello world").expect("Encoding failed");
        assert!(!ids.is_empty());
        assert!(tokenizer.vocab_size() > 0);
    }
}
