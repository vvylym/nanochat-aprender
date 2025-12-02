//! Integration tests for forward pass

use nanochat_model::{GPT, GPTConfig, attention::KVCache};
use aprender::autograd::Tensor;

#[test]
fn test_forward_pass_basic() {
    // Test basic forward pass through the entire model
    let config = GPTConfig::default();
    let mut model = GPT::new(config);
    
    // Create input tokens [batch=1, seq_len=5]
    let idx = Tensor::zeros(&[1, 5]);
    
    // Forward pass without targets (inference mode)
    let logits = model.forward_cache(&idx, None).unwrap();
    
    // Verify output shape: [batch, seq_len, vocab_size]
    let shape = logits.shape();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], 1); // batch
    assert_eq!(shape[1], 5); // seq_len
    assert_eq!(shape[2], 50304); // vocab_size
}

#[test]
fn test_forward_pass_with_kv_cache() {
    // Test forward pass with KV cache for inference
    let config = GPTConfig::default();
    let mut model = GPT::new(config);
    
    // Create input tokens [batch=1, seq_len=1] for autoregressive generation
    let idx = Tensor::zeros(&[1, 1]);
    let mut kv_cache = KVCache::new();
    
    // First forward pass - should populate KV cache
    let logits1 = model.forward_cache(&idx, Some(&mut kv_cache)).unwrap();
    
    // Verify output shape
    let shape1 = logits1.shape();
    assert_eq!(shape1, &[1, 1, 50304]);
    
    // Second forward pass with same cache - should reuse cached values
    let logits2 = model.forward_cache(&idx, Some(&mut kv_cache)).unwrap();
    
    // Verify output shape
    let shape2 = logits2.shape();
    assert_eq!(shape2, &[1, 1, 50304]);
}

#[test]
fn test_forward_pass_different_sequence_lengths() {
    // Test forward pass with different sequence lengths
    let config = GPTConfig::default();
    let mut model = GPT::new(config);
    
    // Test with short sequence
    let idx_short = Tensor::zeros(&[1, 3]);
    let logits_short = model.forward_cache(&idx_short, None).unwrap();
    assert_eq!(logits_short.shape(), &[1, 3, 50304]);
    
    // Test with longer sequence
    let idx_long = Tensor::zeros(&[1, 10]);
    let logits_long = model.forward_cache(&idx_long, None).unwrap();
    assert_eq!(logits_long.shape(), &[1, 10, 50304]);
}

#[test]
fn test_forward_pass_batch_processing() {
    // Test forward pass with batch processing
    let config = GPTConfig::default();
    let mut model = GPT::new(config);
    
    // Create batch input [batch=2, seq_len=5]
    let idx = Tensor::zeros(&[2, 5]);
    
    let logits = model.forward_cache(&idx, None).unwrap();
    
    // Verify output shape: [batch, seq_len, vocab_size]
    assert_eq!(logits.shape(), &[2, 5, 50304]);
}

#[test]
fn test_forward_pass_output_shape() {
    // Test that forward pass output has correct shape [batch, seq_len, vocab_size]
    let config = GPTConfig::default();
    let mut model = GPT::new(config);
    
    let batch = 3;
    let seq_len = 7;
    let idx = Tensor::zeros(&[batch, seq_len]);
    
    let logits = model.forward_cache(&idx, None).unwrap();
    
    // Verify vocab size matches config
    assert_eq!(model.config().vocab_size, 50304);
    assert_eq!(logits.shape(), &[batch, seq_len, 50304]);
}

#[test]
fn test_forward_pass_autoregressive() {
    // Test autoregressive generation (single token at a time)
    let config = GPTConfig::default();
    let mut model = GPT::new(config);
    
    // Simulate autoregressive generation: generate one token at a time
    let mut kv_cache = KVCache::new();
    
    // First token
    let idx1 = Tensor::zeros(&[1, 1]);
    let logits1 = model.forward_cache(&idx1, Some(&mut kv_cache)).unwrap();
    assert_eq!(logits1.shape(), &[1, 1, 50304]);
    
    // Second token (reusing cache)
    let idx2 = Tensor::zeros(&[1, 1]);
    let logits2 = model.forward_cache(&idx2, Some(&mut kv_cache)).unwrap();
    assert_eq!(logits2.shape(), &[1, 1, 50304]);
    
    // Third token (reusing cache)
    let idx3 = Tensor::zeros(&[1, 1]);
    let logits3 = model.forward_cache(&idx3, Some(&mut kv_cache)).unwrap();
    assert_eq!(logits3.shape(), &[1, 1, 50304]);
}
