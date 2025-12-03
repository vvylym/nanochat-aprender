//! Unit tests for ModelConfig validation

use nanochat_model::{config::ConfigError, GPTConfig};

#[test]
fn test_config_default() {
    let config = GPTConfig::default();

    assert_eq!(config.sequence_len, 1024);
    assert_eq!(config.vocab_size, 50304);
    assert_eq!(config.n_layer, 12);
    assert_eq!(config.n_head, 6);
    assert_eq!(config.n_kv_head, 6);
    assert_eq!(config.n_embd, 768);
    assert_eq!(config.n_embd % config.n_head, 0);
}

#[test]
fn test_config_validation_n_kv_head_less_than_or_equal_n_head() {
    let config = GPTConfig::default();

    // n_kv_head should be <= n_head
    assert!(config.n_kv_head <= config.n_head);
}

#[test]
fn test_config_validation_n_head_divisible_by_n_kv_head() {
    let config = GPTConfig::default();

    // n_head should be divisible by n_kv_head for GQA
    assert_eq!(config.n_head % config.n_kv_head, 0);
}

#[test]
fn test_config_custom_values() {
    let config = GPTConfig {
        sequence_len: 2048,
        vocab_size: 100000,
        n_layer: 24,
        n_head: 16,
        n_kv_head: 8,
        n_embd: 1024,
        dropout: Some(0.1),
        seed: None,
    };

    assert_eq!(config.sequence_len, 2048);
    assert_eq!(config.vocab_size, 100000);
    assert_eq!(config.n_layer, 24);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.n_kv_head, 8);
    assert_eq!(config.n_embd, 1024);
    assert_eq!(config.dropout, Some(0.1));
    assert_eq!(config.seed, None);

    // Validate divisibility
    assert_eq!(config.n_embd % config.n_head, 0);
    assert!(config.n_kv_head <= config.n_head);
    assert_eq!(config.n_head % config.n_kv_head, 0);
}

#[test]
fn test_config_with_seed() {
    let config1 = GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), Some(42))
        .expect("Failed to create config");
    let config2 = GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), Some(42))
        .expect("Failed to create config");

    assert_eq!(config1.seed, Some(42));
    assert_eq!(config2.seed, Some(42));
    assert_eq!(config1.seed, config2.seed);
}

#[test]
fn test_config_seed_none() {
    let config = GPTConfig::default();
    assert_eq!(config.seed, None);

    let config_with_seed = GPTConfig::new(1024, 50304, 12, 6, 6, 768, Some(0.0), Some(123))
        .expect("Failed to create config");
    assert_eq!(config_with_seed.seed, Some(123));
}

#[test]
fn test_validate_vocab_size_match() {
    let config =
        GPTConfig::new(1024, 500, 12, 6, 6, 768, Some(0.0), None).expect("Failed to create config");
    assert!(config.validate_vocab_size(500).is_ok());
}

#[test]
fn test_validate_vocab_size_mismatch() {
    let config =
        GPTConfig::new(1024, 500, 12, 6, 6, 768, Some(0.0), None).expect("Failed to create config");
    let result = config.validate_vocab_size(600);
    assert!(result.is_err());

    if let Err(ConfigError::VocabSizeMismatch {
        config: c,
        tokenizer: t,
    }) = result
    {
        assert_eq!(c, 500);
        assert_eq!(t, 600);
    } else {
        panic!("Expected VocabSizeMismatch error");
    }
}

#[test]
fn test_with_tokenizer_vocab_size() {
    let tokenizer_vocab_size = 1000;
    let config = GPTConfig::with_tokenizer_vocab_size(
        tokenizer_vocab_size,
        1024,
        12,
        6,
        6,
        768,
        Some(0.0),
        None,
    )
    .expect("Failed to create config");

    assert_eq!(config.vocab_size, tokenizer_vocab_size);
    assert!(config.validate_vocab_size(tokenizer_vocab_size).is_ok());
}
