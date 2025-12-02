//! Unit tests for attention mechanism

use aprender::autograd::Tensor;
use nanochat_model::attention::{apply_qk_norm, CausalSelfAttention, KVCache};

#[test]
fn test_attention_basic() {
    // Test basic attention computation
    let attn = CausalSelfAttention::new(768, 6, 6, None, None);
    let x = Tensor::ones(&[1, 10, 768]);

    let output = attn.forward(&x, None, 0, None).unwrap();

    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_attention_gqa() {
    // Test GroupedQueryAttention with n_kv_head < n_head
    let attn = CausalSelfAttention::new(768, 6, 2, None, None);
    let x = Tensor::ones(&[1, 10, 768]);

    let output = attn.forward(&x, None, 0, None).unwrap();

    assert_eq!(output.shape(), x.shape());
    assert_eq!(attn.n_head(), 6);
    assert_eq!(attn.n_kv_head(), 2);
}

#[test]
fn test_attention_qk_norm() {
    // Test QK normalization (normalize queries and keys after RoPE)
    let q = Tensor::ones(&[1, 2, 3, 4]);
    let k = Tensor::ones(&[1, 2, 3, 4]);

    let (q_norm, k_norm) = apply_qk_norm(&q, &k).unwrap();

    assert_eq!(q_norm.shape(), q.shape());
    assert_eq!(k_norm.shape(), k.shape());
}

#[test]
fn test_attention_causal_mask() {
    // Test causal attention masking for auto-regressive generation
    let attn = CausalSelfAttention::new(768, 6, 6, None, None);
    let x = Tensor::ones(&[1, 10, 768]);

    // Forward pass should work with causal masking (handled internally)
    let output = attn.forward(&x, None, 0, None).unwrap();

    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_attention_kv_cache() {
    // Test KV cache support for efficient inference
    let attn = CausalSelfAttention::new(768, 6, 6, None, None);
    let x = Tensor::ones(&[1, 10, 768]);
    let mut kv_cache = KVCache::new();

    // Forward pass with KV cache
    let output = attn.forward(&x, Some(&mut kv_cache), 0, None).unwrap();

    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_attention_different_head_counts() {
    // Test attention with different n_head and n_kv_head configurations
    let attn1 = CausalSelfAttention::new(768, 8, 8, None, None);
    let attn2 = CausalSelfAttention::new(768, 8, 2, None, None);
    let attn3 = CausalSelfAttention::new(768, 8, 4, None, None);

    let x = Tensor::ones(&[1, 10, 768]);

    let out1 = attn1.forward(&x, None, 0, None).unwrap();
    let out2 = attn2.forward(&x, None, 0, None).unwrap();
    let out3 = attn3.forward(&x, None, 0, None).unwrap();

    assert_eq!(out1.shape(), x.shape());
    assert_eq!(out2.shape(), x.shape());
    assert_eq!(out3.shape(), x.shape());
}

#[test]
fn test_attention_sequence_lengths() {
    // Test attention with different sequence lengths
    let attn = CausalSelfAttention::new(768, 6, 6, None, None);

    for seq_len in [1, 5, 10, 20, 50] {
        let x = Tensor::ones(&[1, seq_len, 768]);
        let output = attn.forward(&x, None, 0, None).unwrap();
        assert_eq!(output.shape(), x.shape());
    }
}
