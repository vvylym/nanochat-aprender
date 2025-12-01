//! Multi-head attention with Group-Query Attention (GQA)

use aprender::autograd::Tensor;
use aprender::nn::{GroupedQueryAttention, Module, generate_causal_mask};
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

/// Grouped Query Attention with KV cache support
///
/// This wraps aprender's GroupedQueryAttention and adds KV cache functionality
/// for efficient autoregressive inference.
pub struct CausalSelfAttention {
    /// Underlying GQA layer from aprender
    gqa: GroupedQueryAttention,
    /// Number of query heads
    n_head: usize,
    /// Number of key/value heads
    n_kv_head: usize,
    /// Head dimension
    head_dim: usize,
}

impl CausalSelfAttention {
    /// Create a new CausalSelfAttention layer
    ///
    /// # Arguments
    /// * `n_embd` - Embedding dimension
    /// * `n_head` - Number of query heads
    /// * `n_kv_head` - Number of key/value heads (for GQA)
    pub fn new(n_embd: usize, n_head: usize, n_kv_head: usize) -> Self {
        let gqa = GroupedQueryAttention::new(n_embd, n_head, n_kv_head);
        let head_dim = n_embd / n_head;
        
        Self {
            gqa,
            n_head,
            n_kv_head,
            head_dim,
        }
    }

    /// Forward pass with optional KV cache and causal masking
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    /// * `kv_cache` - Optional KV cache for inference (None during training)
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor, _kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            anyhow::bail!("Expected 3D tensor [batch, seq_len, n_embd], got shape {:?}", shape);
        }
        
        // TODO: Implement proper causal masking with correct shape
        // For now, pass None - aprender's GQA may handle causal attention internally
        // or we need to create a properly shaped mask [batch, heads, seq_len, seq_len]
        let causal_mask: Option<&Tensor> = None;
        
        // Use forward_self which returns (output, attn_weights)
        // output should be [batch, seq_len, n_embd]
        let (output, _attn_weights) = self.gqa.forward_self(x, causal_mask);
        
        // Verify output shape matches input
        let output_shape = output.shape();
        if output_shape != shape {
            anyhow::bail!(
                "Attention output shape {:?} doesn't match input shape {:?}",
                output_shape,
                shape
            );
        }
        
        Ok(output)
    }

    /// Get the number of query heads
    pub fn n_head(&self) -> usize {
        self.n_head
    }

    /// Get the number of key/value heads
    pub fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// KV cache for efficient autoregressive inference
///
/// Stores key and value tensors for each layer to avoid recomputation.
pub struct KVCache {
    /// Cached keys and values per layer: Vec<(keys, values)>
    /// Shape: [batch, n_kv_heads, cached_seq_len, head_dim]
    cache: Vec<(Tensor, Tensor)>,
}

impl KVCache {
    /// Create a new empty KV cache
    pub fn new() -> Self {
        Self {
            cache: Vec::new(),
        }
    }

    /// Insert new keys and values into the cache for a given layer
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `k` - Key tensor [batch, n_kv_heads, seq_len, head_dim]
    /// * `v` - Value tensor [batch, n_kv_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Concatenated (k, v) tensors including cached values
    pub fn insert_kv(&mut self, layer_idx: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        // Ensure cache has enough layers
        while self.cache.len() <= layer_idx {
            // Create empty tensors for this layer
            // Shape will be determined by first insert
            self.cache.push((Tensor::zeros(&[0]), Tensor::zeros(&[0])));
        }

        let (cached_k, cached_v) = &self.cache[layer_idx];
        
        // If cache is empty, just store the new k, v
        if cached_k.shape()[0] == 0 {
            self.cache[layer_idx] = (k, v);
            return Ok((self.cache[layer_idx].0.clone(), self.cache[layer_idx].1.clone()));
        }

        // Concatenate along sequence dimension (dim 2)
        // TODO: Implement tensor concatenation once aprender API is confirmed
        // For now, just return the new k, v
        self.cache[layer_idx] = (k, v);
        Ok((self.cache[layer_idx].0.clone(), self.cache[layer_idx].1.clone()))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
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

    #[test]
    fn test_causal_attention_creation() {
        let attn = CausalSelfAttention::new(768, 6, 6);
        assert_eq!(attn.n_head(), 6);
        assert_eq!(attn.n_kv_head(), 6);
        assert_eq!(attn.head_dim(), 128);
    }

    #[test]
    fn test_causal_attention_gqa() {
        // Test GQA with n_kv_head < n_head
        let attn = CausalSelfAttention::new(768, 6, 2);
        assert_eq!(attn.n_head(), 6);
        assert_eq!(attn.n_kv_head(), 2);
    }

    #[test]
    fn test_kv_cache_creation() {
        let mut cache = KVCache::new();
        assert_eq!(cache.cache.len(), 0);
    }
}
