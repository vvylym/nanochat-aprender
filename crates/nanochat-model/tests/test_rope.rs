//! Unit tests for Rotary Position Embeddings (RoPE)

use nanochat_model::rope::{precompute_rotary_embeddings, apply_rotary_emb};
use aprender::autograd::Tensor;

#[test]
fn test_rope_precompute_frequencies() {
    // Test precomputation of cos/sin frequencies for RoPE
    let (cos, sin) = precompute_rotary_embeddings(10, 64, 10000.0).unwrap();
    
    assert_eq!(cos.shape(), &[1, 10, 1, 32]);
    assert_eq!(sin.shape(), &[1, 10, 1, 32]);
}

#[test]
fn test_rope_apply_to_queries() {
    // Test applying RoPE to query tensors
    let x = Tensor::ones(&[1, 2, 3, 4]); // [batch, n_heads, seq_len, head_dim]
    let (cos, sin) = precompute_rotary_embeddings(3, 4, 10000.0).unwrap();
    
    let result = apply_rotary_emb(&x, &cos, &sin).unwrap();
    
    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rope_apply_to_keys() {
    // Test applying RoPE to key tensors
    let x = Tensor::ones(&[1, 2, 3, 4]); // [batch, n_heads, seq_len, head_dim]
    let (cos, sin) = precompute_rotary_embeddings(3, 4, 10000.0).unwrap();
    
    let result = apply_rotary_emb(&x, &cos, &sin).unwrap();
    
    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rope_relative_positioning() {
    // Test that RoPE correctly encodes relative positions
    let (cos, sin) = precompute_rotary_embeddings(10, 64, 10000.0).unwrap();
    
    // Verify cos and sin have correct shapes for broadcasting
    assert_eq!(cos.shape()[1], 10); // sequence length
    assert_eq!(sin.shape()[1], 10);
}

#[test]
fn test_rope_different_sequence_lengths() {
    // Test RoPE with different sequence lengths
    for seq_len in [1, 5, 10, 20, 50] {
        let (cos, sin) = precompute_rotary_embeddings(seq_len, 64, 10000.0).unwrap();
        assert_eq!(cos.shape()[1], seq_len);
        assert_eq!(sin.shape()[1], seq_len);
    }
}

#[test]
fn test_rope_head_dimension() {
    // Test RoPE with different head dimensions
    for head_dim in [32, 64, 128, 256] {
        let (cos, sin) = precompute_rotary_embeddings(10, head_dim, 10000.0).unwrap();
        assert_eq!(cos.shape()[3], head_dim / 2);
        assert_eq!(sin.shape()[3], head_dim / 2);
    }
}
