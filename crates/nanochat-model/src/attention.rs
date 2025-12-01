//! Multi-head attention with Group-Query Attention (GQA)

use aprender::autograd::Tensor;
use crate::norm::rms_norm;
use anyhow::Result;

/// Apply QK normalization to queries and keys
///
/// This normalizes queries and keys after applying RoPE, as per the nanochat architecture.
/// QK normalization helps stabilize training and improve model performance.
///
/// # Arguments
/// * `q` - Query tensor of shape [batch, n_heads, seq_len, head_dim]
/// * `k` - Key tensor of shape [batch, n_kv_heads, seq_len, head_dim]
///
/// # Returns
/// Tuple of (normalized_q, normalized_k)
pub fn apply_qk_norm(q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
    // Apply RMSNorm to queries and keys
    // This is done after RoPE in the attention mechanism
    let q_norm = rms_norm(q)?;
    let k_norm = rms_norm(k)?;
    
    Ok((q_norm, k_norm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qk_norm_basic() {
        // Create query and key tensors
        let q = Tensor::ones(&[1, 2, 3, 4]);
        let k = Tensor::ones(&[1, 2, 3, 4]);
        
        let (q_norm, k_norm) = apply_qk_norm(&q, &k).unwrap();
        
        assert_eq!(q_norm.shape(), q.shape());
        assert_eq!(k_norm.shape(), k.shape());
    }

    #[test]
    fn test_qk_norm_different_shapes() {
        // Test with different n_heads vs n_kv_heads (GQA)
        let q = Tensor::ones(&[1, 4, 3, 4]); // 4 query heads
        let k = Tensor::ones(&[1, 2, 3, 4]);  // 2 key/value heads
        
        let (q_norm, k_norm) = apply_qk_norm(&q, &k).unwrap();
        
        assert_eq!(q_norm.shape(), q.shape());
        assert_eq!(k_norm.shape(), k.shape());
    }
}
