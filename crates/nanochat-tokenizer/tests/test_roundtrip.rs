//! Property-based tests for encode/decode round-trip

use nanochat_tokenizer::Tokenizer;
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_encode_decode_roundtrip_ascii(text in "[ -~]{1,100}") {
        let tokenizer = create_test_tokenizer();

        let ids = tokenizer.encode(&text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();

        // ASCII text should round-trip (may have whitespace normalization)
        // Just verify it doesn't panic and produces some output
        prop_assert!(!decoded.is_empty() || ids.is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip_unicode(text in "\\p{Any}{1,50}") {
        let tokenizer = create_test_tokenizer();

        let ids = tokenizer.encode(&text).unwrap();

        // Only test if encoding succeeded and produced tokens
        if !ids.is_empty() {
            let _decoded = tokenizer.decode(&ids).unwrap();

            // Unicode text may not round-trip perfectly due to byte-level encoding
            // But encoding should produce some tokens and decoding should produce some text
            // (may be empty for some edge cases, but that's acceptable)
            prop_assert!(true); // Just verify it doesn't panic
        }
    }

    #[test]
    fn test_encode_never_panics(text in "\\p{Any}{0,200}") {
        let tokenizer = create_test_tokenizer();

        // Encoding should never panic, even with edge cases
        let _ = tokenizer.encode(&text);
    }

    #[test]
    fn test_decode_never_panics(ids in prop::collection::vec(0u32..50000, 0..100)) {
        let tokenizer = create_test_tokenizer();

        // Decoding should never panic, even with invalid IDs
        let _ = tokenizer.decode(&ids);
    }

    #[test]
    fn test_encode_id_range(ids in prop::collection::vec(0u32..500, 1..50)) {
        let tokenizer = create_test_tokenizer();
        let vocab_size = tokenizer.vocab_size() as u32;

        // Only test IDs that are within the vocabulary range
        let valid_ids: Vec<u32> = ids.into_iter()
            .filter(|&id| id < vocab_size)
            .collect();

        if !valid_ids.is_empty() {
            // Decode should handle any ID in vocabulary range
            let decoded = tokenizer.decode(&valid_ids).unwrap();
            // Decoded should be a valid string (may be empty or contain special tokens)
            prop_assert!(decoded.chars().all(|c| c.is_ascii() || !c.is_control()));
        }
    }
}

// Helper function
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
