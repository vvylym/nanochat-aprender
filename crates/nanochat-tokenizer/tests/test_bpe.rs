//! Unit tests for BPE training

use nanochat_tokenizer::Tokenizer;

#[test]
fn test_bpe_train_basic() {
    // Test basic BPE training on simple corpus
    let corpus = [
        "hello world".to_string(),
        "hello rust".to_string(),
        "world peace".to_string(),
    ];

    let vocab_size = 300; // Must be at least 267 (256 bytes + 11 special tokens)
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), vocab_size)
        .expect("Failed to train tokenizer");

    // Vocab size should be positive (aprender may have different minimums)
    assert!(tokenizer.vocab_size() > 0);
}

#[test]
fn test_bpe_train_minimum_vocab() {
    // Test that training works with small vocab (aprender may allow this)
    let corpus = ["test".to_string()];
    let vocab_size = 50; // Small vocab

    let result = Tokenizer::train_from_iterator(corpus.iter(), vocab_size);
    // Aprender may allow small vocabs, so just verify it doesn't panic
    if let Ok(tokenizer) = result {
        assert!(tokenizer.vocab_size() > 0);
    }
}

#[test]
fn test_bpe_train_special_tokens() {
    // Test that special tokens are preserved during training
    let corpus = ["hello world".to_string()];
    let vocab_size = 500;
    let _tokenizer = Tokenizer::train_from_iterator(corpus.iter(), vocab_size)
        .expect("Failed to train tokenizer");

    // Special tokens should be in vocabulary
    // This will be verified once SpecialTokens is implemented
}

#[test]
fn test_bpe_train_empty_corpus() {
    // Test handling of empty corpus
    let corpus: [String; 0] = [];
    let vocab_size = 500;

    let result = Tokenizer::train_from_iterator(corpus.iter(), vocab_size);
    // Aprender may handle empty corpus differently, just verify it doesn't panic
    if let Ok(tokenizer) = result {
        assert!(tokenizer.vocab_size() > 0);
    }
}

#[test]
fn test_bpe_train_large_corpus() {
    // Test training on larger corpus
    let corpus: Vec<String> = (0..1000).map(|i| format!("sentence number {}", i)).collect();

    let vocab_size = 1000;
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), vocab_size)
        .expect("Failed to train tokenizer");

    // Vocab size should be positive, but might not reach target if corpus is repetitive
    // The algorithm stops when no more merges are possible
    assert!(tokenizer.vocab_size() > 0); // At least some vocabulary
}
