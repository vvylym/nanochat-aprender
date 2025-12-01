//! Unit tests for special token handling

use nanochat_tokenizer::{SpecialTokens, Tokenizer};

#[test]
fn test_special_tokens_definition() {
    let special = SpecialTokens::default();
    
    assert!(!special.bos().is_empty());
    assert!(!special.eos().is_empty());
    assert!(!special.pad().is_empty());
}

#[test]
fn test_bos_token() {
    let special = SpecialTokens::default();
    let bos = special.bos();
    
    assert_eq!(bos, "<|bos|>");
}

#[test]
fn test_encode_with_bos() {
    let tokenizer = create_test_tokenizer();
    let text = "hello";
    
    let ids = tokenizer.encode_with_special(text, Some("<|bos|>"), None).unwrap();
    assert!(!ids.is_empty());
    // First token should be BOS
    // This will be verified once implementation is complete
}

#[test]
fn test_encode_with_eos() {
    let tokenizer = create_test_tokenizer();
    let text = "hello";
    
    let ids = tokenizer.encode_with_special(text, None, Some("<|eos|>")).unwrap();
    assert!(!ids.is_empty());
    // Last token should be EOS
}

#[test]
fn test_special_token_ids() {
    let tokenizer = create_test_tokenizer();
    
    let bos_id = tokenizer.special_token_id("<|bos|>").unwrap();
    let eos_id = tokenizer.special_token_id("<|eos|>").unwrap();
    
    assert_ne!(bos_id, eos_id);
}

#[test]
fn test_decode_preserves_special_tokens() {
    let tokenizer = create_test_tokenizer();
    let text = "hello";
    
    let ids = tokenizer.encode_with_special(text, Some("<|bos|>"), Some("<|eos|>")).unwrap();
    
    // Special tokens should be in the encoded IDs
    let bos_id = tokenizer.special_token_id("<|bos|>").unwrap();
    let eos_id = tokenizer.special_token_id("<|eos|>").unwrap();
    
    // Verify special token IDs are present in encoded output
    assert!(ids.contains(&bos_id), "BOS token ID should be in encoded output");
    assert!(ids.contains(&eos_id), "EOS token ID should be in encoded output");
    
    // Decoded text should not contain special tokens (they are metadata)
    let decoded = tokenizer.decode(&ids).unwrap();
    // Decoded text should contain the original text
    assert!(decoded.contains("hello") || decoded.trim() == "hello");
}

// Helper function
fn create_test_tokenizer() -> Tokenizer {
    // Create a small tokenizer for testing
    let corpus = vec![
        "hello world",
        "hello rust",
        "world peace",
    ];
    Tokenizer::train_from_iterator(corpus.iter(), 500)
        .expect("Failed to create test tokenizer")
}

