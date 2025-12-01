//! Vocabulary management

use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during vocabulary operations
#[derive(Debug, Error)]
pub enum VocabularyError {
    #[error("Token not found in vocabulary: {0}")]
    TokenNotFound(String),
    #[error("ID not found in vocabulary: {0}")]
    IdNotFound(u32),
    #[error("Vocabulary is empty")]
    EmptyVocabulary,
}

/// Vocabulary mapping between tokens and IDs
///
/// Maintains bidirectional mappings:
/// - token -> ID (for encoding)
/// - ID -> token (for decoding)
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Mapping from token to ID
    token_to_id: HashMap<String, u32>,
    /// Mapping from ID to token
    id_to_token: HashMap<u32, String>,
    /// Next available ID for new tokens
    next_id: u32,
}

impl Vocabulary {
    /// Create a new empty vocabulary
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a token to the vocabulary
    ///
    /// Returns the ID assigned to the token. If the token already exists,
    /// returns its existing ID.
    pub fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }

        let id = self.next_id;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        self.next_id += 1;
        id
    }

    /// Get the ID for a token
    ///
    /// Returns an error if the token is not in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Result<u32, VocabularyError> {
        self.token_to_id
            .get(token)
            .copied()
            .ok_or_else(|| VocabularyError::TokenNotFound(token.to_string()))
    }

    /// Get the token for an ID
    ///
    /// Returns an error if the ID is not in the vocabulary.
    pub fn id_to_token(&self, id: u32) -> Result<&str, VocabularyError> {
        self.id_to_token
            .get(&id)
            .map(|s| s.as_str())
            .ok_or(VocabularyError::IdNotFound(id))
    }

    /// Check if a token exists in the vocabulary
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Check if an ID exists in the vocabulary
    pub fn contains_id(&self, id: u32) -> bool {
        self.id_to_token.contains_key(&id)
    }

    /// Get the size of the vocabulary
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Check if the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Get all tokens in the vocabulary
    pub fn tokens(&self) -> impl Iterator<Item = &String> {
        self.token_to_id.keys()
    }

    /// Get all IDs in the vocabulary
    pub fn ids(&self) -> impl Iterator<Item = &u32> {
        self.id_to_token.keys()
    }

    /// Initialize vocabulary with byte-level tokens (0-255)
    ///
    /// This is the base vocabulary for BPE, containing all possible byte values.
    pub fn init_byte_level(&mut self) {
        for byte in 0u8..=255u8 {
            let token = format!("<0x{:02x}>", byte);
            self.add_token(token);
        }
    }

    /// Add special tokens to the vocabulary
    ///
    /// Special tokens are added after byte-level tokens but before BPE merges.
    pub fn add_special_tokens(&mut self, special_tokens: &[String]) {
        for token in special_tokens {
            self.add_token(token.clone());
        }
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_add_token() {
        let mut vocab = Vocabulary::new();
        let id = vocab.add_token("hello".to_string());
        assert_eq!(id, 0);
        assert_eq!(vocab.size(), 1);
    }

    #[test]
    fn test_vocab_duplicate_token() {
        let mut vocab = Vocabulary::new();
        let id1 = vocab.add_token("hello".to_string());
        let id2 = vocab.add_token("hello".to_string());
        assert_eq!(id1, id2);
        assert_eq!(vocab.size(), 1);
    }

    #[test]
    fn test_vocab_token_to_id() {
        let mut vocab = Vocabulary::new();
        let id = vocab.add_token("hello".to_string());
        assert_eq!(vocab.token_to_id("hello").unwrap(), id);
    }

    #[test]
    fn test_vocab_id_to_token() {
        let mut vocab = Vocabulary::new();
        let id = vocab.add_token("hello".to_string());
        assert_eq!(vocab.id_to_token(id).unwrap(), "hello");
    }

    #[test]
    fn test_vocab_init_byte_level() {
        let mut vocab = Vocabulary::new();
        vocab.init_byte_level();
        assert_eq!(vocab.size(), 256);
        assert!(vocab.contains_token("<0x00>"));
        assert!(vocab.contains_token("<0xff>"));
    }
}
