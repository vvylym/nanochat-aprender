//! Byte Pair Encoding (BPE) implementation

use std::collections::HashMap;
use anyhow::{Context, Result};
use regex::Regex;
use crate::vocab::Vocabulary;
use crate::special_tokens::SpecialTokens;

/// GPT-4 style split pattern for pre-tokenization
/// This pattern splits text into groups before BPE is applied
/// Note: Simplified from original to work with Rust regex (no look-ahead)
const SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+";

/// BPE tokenizer implementation
///
/// This implements Byte Pair Encoding (BPE) as used in GPT-4 style tokenizers.
/// The algorithm:
/// 1. Start with byte-level vocabulary (256 tokens)
/// 2. Pre-tokenize text using regex pattern
/// 3. Count all pairs of adjacent tokens
/// 4. Find and merge the most frequent pair
/// 5. Repeat until desired vocabulary size
#[derive(Debug, Clone)]
pub struct BPE {
    /// Vocabulary mapping
    vocab: Vocabulary,
    /// BPE merges: (token1, token2) -> merged_token
    merges: Vec<(String, String)>,
    /// Pre-tokenization regex pattern
    split_pattern: Regex,
    /// Special tokens
    special_tokens: SpecialTokens,
}

impl BPE {
    /// Train a BPE tokenizer from an iterator of text
    ///
    /// # Arguments
    /// * `text_iterator` - Iterator over training text
    /// * `vocab_size` - Target vocabulary size (must be >= 256 + number of special tokens)
    ///
    /// # Returns
    /// A trained BPE tokenizer
    pub fn train_from_iterator<I, S>(text_iterator: I, vocab_size: usize) -> Result<Self>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let special_tokens = SpecialTokens::default();
        let num_special = special_tokens.all().len();
        
        // Validate vocab size
        if vocab_size < 256 + num_special {
            anyhow::bail!(
                "vocab_size must be at least {} (256 bytes + {} special tokens), got {}",
                256 + num_special,
                num_special,
                vocab_size
            );
        }

        let vocab_size_no_special = vocab_size - num_special;

        // Initialize vocabulary with byte-level tokens
        let mut vocab = Vocabulary::new();
        vocab.init_byte_level();

        // Add special tokens
        vocab.add_special_tokens(&special_tokens.all().iter().map(|s| s.to_string()).collect::<Vec<_>>());

        // Pre-tokenize all text
        let split_pattern = Regex::new(SPLIT_PATTERN)
            .context("Failed to compile split pattern regex")?;
        
        let mut word_freqs: HashMap<Vec<u32>, usize> = HashMap::new();
        
        // Collect and pre-tokenize all text
        for text in text_iterator {
            let text = text.as_ref();
            if text.is_empty() {
                continue;
            }
            
            // Pre-tokenize using regex
            let words = Self::pre_tokenize(&split_pattern, text);
            
            for word in words {
                // Convert word to byte-level tokens
                let word_bytes = word.as_bytes();
                let mut word_tokens = Vec::new();
                
                for &byte in word_bytes {
                    let token = format!("<0x{:02x}>", byte);
                    let id = vocab.token_to_id(&token)
                        .unwrap_or_else(|_| vocab.add_token(token));
                    word_tokens.push(id);
                }
                
                *word_freqs.entry(word_tokens).or_insert(0) += 1;
            }
        }

        if word_freqs.is_empty() {
            anyhow::bail!("Empty corpus: no text found for training");
        }

        // Perform BPE merges
        let mut merges = Vec::new();
        let mut vocab_size_current = vocab.size();

        while vocab_size_current < vocab_size_no_special {
            // Count all pairs
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            
            for (word, &freq) in &word_freqs {
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i], word[i + 1]);
                    *pair_counts.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_counts.is_empty() {
                break; // No more pairs to merge
            }

            // Find most frequent pair
            let (&best_pair, _) = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .expect("pair_counts is not empty");

            // Create merged token - get token strings first
            let token1_str = vocab.id_to_token(best_pair.0)
                .context("Token ID not found in vocabulary")?
                .to_string();
            let token2_str = vocab.id_to_token(best_pair.1)
                .context("Token ID not found in vocabulary")?
                .to_string();
            let merged_token = format!("{}{}", token1_str, token2_str);
            
            // Add merged token to vocabulary
            let merged_id = vocab.add_token(merged_token.clone());
            merges.push((token1_str, token2_str));

            // Update word_freqs: replace all occurrences of the pair
            let mut new_word_freqs = HashMap::new();
            for (word, freq) in word_freqs {
                let new_word = Self::apply_merge(&word, best_pair.0, best_pair.1, merged_id);
                *new_word_freqs.entry(new_word).or_insert(0) += freq;
            }
            word_freqs = new_word_freqs;

            vocab_size_current = vocab.size();
        }

        Ok(Self {
            vocab,
            merges,
            split_pattern,
            special_tokens,
        })
    }

    /// Pre-tokenize text using regex pattern
    fn pre_tokenize(pattern: &Regex, text: &str) -> Vec<String> {
        pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Apply a BPE merge to a word
    fn apply_merge(word: &[u32], id1: u32, id2: u32, merged_id: u32) -> Vec<u32> {
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < word.len() {
            if i < word.len().saturating_sub(1) && word[i] == id1 && word[i + 1] == id2 {
                result.push(merged_id);
                i += 2;
            } else {
                result.push(word[i]);
                i += 1;
            }
        }
        
        result
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Pre-tokenize
        let words = Self::pre_tokenize(&self.split_pattern, text);
        let mut all_ids = Vec::new();

        for word in words {
            // Convert word to byte-level tokens
            let word_bytes = word.as_bytes();
            let mut word_tokens = Vec::new();
            
            for &byte in word_bytes {
                let token = format!("<0x{:02x}>", byte);
                let id = self.vocab.token_to_id(&token)
                    .context("Byte token not found in vocabulary")?;
                word_tokens.push(id);
            }

            // Apply all BPE merges
            for (token1, token2) in &self.merges {
                let id1 = self.vocab.token_to_id(token1).ok();
                let id2 = self.vocab.token_to_id(token2).ok();
                let merged_token = format!("{}{}", token1, token2);
                let merged_id = self.vocab.token_to_id(&merged_token).ok();

                if let (Some(id1), Some(id2), Some(merged_id)) = (id1, id2, merged_id) {
                    word_tokens = Self::apply_merge(&word_tokens, id1, id2, merged_id);
                }
            }

            all_ids.extend(word_tokens);
        }

        Ok(all_ids)
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    /// * `ids` - Slice of token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Errors
    /// Returns an error if any token ID is not found in the vocabulary
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if ids.is_empty() {
            return Ok(String::new());
        }

        let mut result = String::new();
        
        for &id in ids {
            let token = self.vocab.id_to_token(id)
                .with_context(|| format!("Token ID {} not found in vocabulary", id))?;
            
            // Handle byte-level tokens
            if token.starts_with("<0x") && token.ends_with(">") && token.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                    // Only push valid UTF-8 characters
                    if byte.is_ascii() || (byte >= 0xC0 && byte <= 0xF4) {
                        // Try to decode as UTF-8
                        let mut buf = [0u8; 4];
                        buf[0] = byte;
                        if let Ok(ch) = std::str::from_utf8(&buf[..1]) {
                            result.push_str(ch);
                        }
                    }
                }
            } else if !self.special_tokens.is_special(token) {
                // Regular merged token - decode bytes
                // Merged tokens are stored as concatenated byte tokens, need to extract bytes
                let bytes = Self::extract_bytes_from_token(token);
                if let Ok(text) = String::from_utf8(bytes) {
                    result.push_str(&text);
                }
            }
            // Special tokens are skipped in decoding (or handled specially)
        }

        Ok(result)
    }

    /// Extract bytes from a merged token string
    fn extract_bytes_from_token(token: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut i = 0;
        
        while i < token.len() {
            if token[i..].starts_with("<0x") && i + 6 <= token.len() {
                if let Ok(byte) = u8::from_str_radix(&token[i+3..i+5], 16) {
                    bytes.push(byte);
                    i += 6;
                    continue;
                }
            }
            i += 1;
        }
        
        bytes
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Get the BPE merges
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }
}
