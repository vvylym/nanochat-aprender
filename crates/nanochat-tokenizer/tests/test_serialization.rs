//! Unit tests for tokenizer serialization/deserialization

use nanochat_tokenizer::Tokenizer;
use std::fs;
use tempfile::TempDir;

fn create_test_tokenizer() -> Tokenizer {
    // Create a small tokenizer for testing
    let corpus = vec![
        "hello world",
        "hello rust",
        "world peace",
        "the quick brown fox",
        "rust is awesome",
    ];
    Tokenizer::train_from_iterator(corpus.iter(), 500).expect("Failed to create test tokenizer")
}

#[test]
fn test_tokenizer_save() {
    let tokenizer = create_test_tokenizer();
    let temp_dir = TempDir::new().unwrap();
    
    tokenizer.save(temp_dir.path()).unwrap();
    
    // Verify tokenizer.json exists
    let tokenizer_file = temp_dir.path().join("tokenizer.json");
    assert!(tokenizer_file.exists(), "tokenizer.json should exist");
    
    // Verify file is valid JSON
    let content = fs::read_to_string(&tokenizer_file).unwrap();
    let _data: serde_json::Value = serde_json::from_str(&content).unwrap();
}

#[test]
fn test_tokenizer_load() {
    let tokenizer1 = create_test_tokenizer();
    let temp_dir = TempDir::new().unwrap();
    
    tokenizer1.save(temp_dir.path()).unwrap();
    
    let tokenizer2 = Tokenizer::from_directory(temp_dir.path()).unwrap();
    
    // Verify vocab_size matches
    assert_eq!(tokenizer1.vocab_size(), tokenizer2.vocab_size());
}

#[test]
fn test_tokenizer_roundtrip() {
    let tokenizer1 = create_test_tokenizer();
    let text = "hello world";
    let ids1 = tokenizer1.encode(text).unwrap();
    
    let temp_dir = TempDir::new().unwrap();
    tokenizer1.save(temp_dir.path()).unwrap();
    
    let tokenizer2 = Tokenizer::from_directory(temp_dir.path()).unwrap();
    let ids2 = tokenizer2.encode(text).unwrap();
    
    // Encoded IDs should match
    assert_eq!(ids1, ids2);
    
    // Decode should also match
    let decoded1 = tokenizer1.decode(&ids1).unwrap();
    let decoded2 = tokenizer2.decode(&ids2).unwrap();
    assert_eq!(decoded1, decoded2);
}

#[test]
fn test_tokenizer_roundtrip_multiple_texts() {
    let tokenizer1 = create_test_tokenizer();
    let texts = vec!["hello", "world", "rust", "the quick brown fox"];
    
    let temp_dir = TempDir::new().unwrap();
    tokenizer1.save(temp_dir.path()).unwrap();
    
    let tokenizer2 = Tokenizer::from_directory(temp_dir.path()).unwrap();
    
    for text in texts {
        let ids1 = tokenizer1.encode(text).unwrap();
        let ids2 = tokenizer2.encode(text).unwrap();
        assert_eq!(ids1, ids2, "Encoded IDs should match for text: {}", text);
    }
}

#[test]
fn test_tokenizer_load_nonexistent_file() {
    let temp_dir = TempDir::new().unwrap();
    
    let result = Tokenizer::from_directory(temp_dir.path());
    assert!(result.is_err(), "Should fail when tokenizer.json doesn't exist");
}

