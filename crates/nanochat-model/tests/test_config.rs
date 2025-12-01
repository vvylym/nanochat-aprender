//! Unit tests for ModelConfig validation

use nanochat_model::GPTConfig;

#[test]
fn test_config_default() {
    let config = GPTConfig::default();
    
    assert_eq!(config.sequence_len, 1024);
    assert_eq!(config.vocab_size, 50304);
    assert_eq!(config.n_layer, 12);
    assert_eq!(config.n_head, 6);
    assert_eq!(config.n_kv_head, 6);
    assert_eq!(config.n_embd, 768);
}

#[test]
fn test_config_validation_n_embd_divisible_by_n_head() {
    let mut config = GPTConfig::default();
    config.n_embd = 768;
    config.n_head = 6;
    
    // 768 / 6 = 128, should be valid
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
        dropout: 0.1,
    };
    
    assert_eq!(config.sequence_len, 2048);
    assert_eq!(config.vocab_size, 100000);
    assert_eq!(config.n_layer, 24);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.n_kv_head, 8);
    assert_eq!(config.n_embd, 1024);
    assert_eq!(config.dropout, 0.1);
    
    // Validate divisibility
    assert_eq!(config.n_embd % config.n_head, 0);
    assert!(config.n_kv_head <= config.n_head);
    assert_eq!(config.n_head % config.n_kv_head, 0);
}

