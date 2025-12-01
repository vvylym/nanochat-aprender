//! Unit tests for BPE training

use nanochat_tokenizer::bpe::BPE;

#[test]
fn test_bpe_train_basic() {
    // Test basic BPE training on simple corpus
    let corpus = vec![
        "hello world".to_string(),
        "hello rust".to_string(),
        "world peace".to_string(),
    ];
    
    let vocab_size = 300; // Must be at least 267 (256 bytes + 11 special tokens)
    let bpe = BPE::train_from_iterator(corpus.iter(), vocab_size).unwrap();
    
    // Vocab size should be at least the minimum (256 + special tokens)
    assert!(bpe.vocab_size() >= 267);
}

#[test]
fn test_bpe_train_minimum_vocab() {
    // Test that training requires at least 256 base tokens
    let corpus = vec!["test".to_string()];
    let vocab_size = 50; // Too small
    
    let result = BPE::train_from_iterator(corpus.iter(), vocab_size);
    assert!(result.is_err());
}

#[test]
fn test_bpe_train_special_tokens() {
    // Test that special tokens are preserved during training
    let corpus = vec!["hello world".to_string()];
    let vocab_size = 500;
    let _bpe = BPE::train_from_iterator(corpus.iter(), vocab_size).unwrap();
    
    // Special tokens should be in vocabulary
    // This will be verified once SpecialTokens is implemented
}

#[test]
fn test_bpe_train_empty_corpus() {
    // Test handling of empty corpus
    let corpus: Vec<String> = vec![];
    let vocab_size = 500;
    
    let result = BPE::train_from_iterator(corpus.iter(), vocab_size);
    assert!(result.is_err());
}

#[test]
fn test_bpe_train_large_corpus() {
    // Test training on larger corpus
    let corpus: Vec<String> = (0..1000)
        .map(|i| format!("sentence number {}", i))
        .collect();
    
    let vocab_size = 1000;
    let bpe = BPE::train_from_iterator(corpus.iter(), vocab_size).unwrap();
    
    // Vocab size should be at least the minimum, but might not reach target if corpus is repetitive
    // The algorithm stops when no more merges are possible
    assert!(bpe.vocab_size() >= 267); // At least base vocabulary
}

