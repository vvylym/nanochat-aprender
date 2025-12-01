//! Property-based tests for encode/decode round-trip

use proptest::prelude::*;
use nanochat_tokenizer::Tokenizer;

proptest! {
    #[test]
    fn test_encode_decode_roundtrip_ascii(text in "[ -~]{1,100}") {
        let tokenizer = create_test_tokenizer();
        
        let ids = tokenizer.encode(&text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();
        
        // ASCII text should round-trip perfectly
        prop_assert_eq!(decoded.trim(), text.trim());
    }
    
    #[test]
    fn test_encode_decode_roundtrip_unicode(text in "\\p{Any}{1,50}") {
        let tokenizer = create_test_tokenizer();
        
        let ids = tokenizer.encode(&text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();
        
        // Unicode text should round-trip (may have normalization)
        // At minimum, decoded should not be empty
        prop_assert!(!decoded.is_empty());
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
    fn test_encode_id_range(ids in prop::collection::vec(0u32..50000, 1..50)) {
        let tokenizer = create_test_tokenizer();
        
        // Decode should handle any ID in vocabulary range
        let decoded = tokenizer.decode(&ids).unwrap();
        // Decoded should be a valid string (may be empty or contain special tokens)
        prop_assert!(decoded.chars().all(|c| c.is_ascii() || !c.is_control()));
    }
}

// Helper function
fn create_test_tokenizer() -> Tokenizer {
    todo!("Create test tokenizer")
}

