//! Unit tests for encoding and decoding

use nanochat_tokenizer::Tokenizer;

#[test]
fn test_encode_basic() {
    let tokenizer = create_test_tokenizer();
    let text = "hello world";
    
    let ids = tokenizer.encode(text).unwrap();
    assert!(!ids.is_empty());
}

#[test]
fn test_decode_basic() {
    let tokenizer = create_test_tokenizer();
    let text = "hello world";
    
    let ids = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&ids).unwrap();
    
    // Decoded text should match original (may have whitespace normalization)
    assert_eq!(decoded.trim(), text.trim());
}

#[test]
fn test_encode_decode_roundtrip() {
    let tokenizer = create_test_tokenizer();
    let texts = vec![
        "hello world",
        "The quick brown fox",
        "Rust is awesome!",
        "123 456 789",
    ];
    
    for text in texts {
        let ids = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();
        assert_eq!(decoded.trim(), text.trim());
    }
}

#[test]
fn test_encode_unicode() {
    let tokenizer = create_test_tokenizer();
    let text = "Hello 世界 🌍";
    
    let ids = tokenizer.encode(text).unwrap();
    assert!(!ids.is_empty());
    
    let decoded = tokenizer.decode(&ids).unwrap();
    assert!(decoded.contains("Hello"));
}

#[test]
fn test_encode_empty_string() {
    let tokenizer = create_test_tokenizer();
    let text = "";
    
    let ids = tokenizer.encode(text).unwrap();
    // Empty string should encode to empty or just special tokens
    assert!(ids.is_empty() || ids.len() <= 1);
}

#[test]
fn test_decode_empty_ids() {
    let tokenizer = create_test_tokenizer();
    let ids = vec![];
    
    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn test_encode_multiple_texts() {
    let tokenizer = create_test_tokenizer();
    let texts = vec!["hello", "world", "rust"];
    
    let all_ids = tokenizer.encode_batch(&texts).unwrap();
    assert_eq!(all_ids.len(), texts.len());
}

// Helper function to create a test tokenizer
fn create_test_tokenizer() -> Tokenizer {
    // Create a small tokenizer for testing
    let corpus = vec![
        "hello world",
        "hello rust",
        "world peace",
        "rust is awesome",
        "the quick brown fox",
    ];
    Tokenizer::train_from_iterator(corpus.iter(), 500)
        .expect("Failed to create test tokenizer")
}

