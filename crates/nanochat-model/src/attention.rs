//! Multi-head attention with Group-Query Attention (GQA)

use crate::norm::rms_norm;
use anyhow::Result;
use aprender::autograd::Tensor;
use aprender::nn::{Dropout, Linear, Module};

// Helper functions to replicate aprender's internal functionality
// These are needed because GQA's internal fields are private

/// Reshape tensor for attention: [batch, seq, embed] -> [batch, heads, seq, head_dim]
fn reshape_for_attention(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    let mut output = vec![0.0; batch * num_heads * seq_len * head_dim];
    let x_data = x.data();

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Input: [b, s, h * head_dim + d]
                    // Output: [b, h, s, d]
                    let in_idx = b * seq_len * (num_heads * head_dim)
                        + s * (num_heads * head_dim)
                        + h * head_dim
                        + d;
                    let out_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    output[out_idx] = x_data[in_idx];
                }
            }
        }
    }

    Tensor::new(&output, &[batch, num_heads, seq_len, head_dim])
}

/// Reshape tensor from attention: [batch, heads, seq, head_dim] -> [batch, seq, embed]
fn reshape_from_attention(x: &Tensor, batch: usize, seq_len: usize, embed_dim: usize) -> Tensor {
    let num_heads = embed_dim / (x.shape()[2] * x.shape()[3] / seq_len);
    let head_dim = embed_dim / num_heads;
    let mut output = vec![0.0; batch * seq_len * embed_dim];
    let x_data = x.data();

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Input: [b, h, s, d]
                    // Output: [b, s, h * head_dim + d]
                    let in_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    let out_idx = b * seq_len * embed_dim + s * embed_dim + h * head_dim + d;
                    output[out_idx] = x_data[in_idx];
                }
            }
        }
    }

    Tensor::new(&output, &[batch, seq_len, embed_dim])
}

/// Repeat KV heads to match Q heads (for Grouped Query Attention)
fn repeat_kv_heads(x: &Tensor, groups: usize) -> Tensor {
    if groups == 1 {
        return x.clone();
    }

    let shape = x.shape();
    let (batch, kv_heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let num_heads = kv_heads * groups;

    let mut output = vec![0.0; batch * num_heads * seq_len * head_dim];
    let x_data = x.data();

    for b in 0..batch {
        for kv_h in 0..kv_heads {
            for g in 0..groups {
                let h = kv_h * groups + g;
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        let in_idx = b * kv_heads * seq_len * head_dim
                            + kv_h * seq_len * head_dim
                            + s * head_dim
                            + d;
                        let out_idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + s * head_dim
                            + d;
                        output[out_idx] = x_data[in_idx];
                    }
                }
            }
        }
    }

    Tensor::new(&output, &[batch, num_heads, seq_len, head_dim])
}

/// Transpose last two dimensions
fn transpose_last_two(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return x.clone();
    }

    let last = shape[ndim - 1];
    let second_last = shape[ndim - 2];

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = last;
    new_shape[ndim - 1] = second_last;

    let batch_size: usize = shape[..ndim - 2].iter().product();
    let matrix_size = last * second_last;

    let mut output = vec![0.0; x.data().len()];
    let x_data = x.data();

    for b in 0..batch_size {
        let offset = b * matrix_size;
        for i in 0..second_last {
            for j in 0..last {
                output[offset + j * second_last + i] = x_data[offset + i * last + j];
            }
        }
    }

    Tensor::new(&output, &new_shape)
}

/// Batched matrix multiplication for 4D tensors
fn matmul_batched_4d(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let (batch, heads, m, k) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
    let n = b_shape[3];

    let mut output = vec![0.0; batch * heads * m * n];
    let a_data = a.data();
    let b_data = b.data();

    for b_idx in 0..batch {
        for h in 0..heads {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        let a_idx = b_idx * heads * m * k + h * m * k + i * k + k_idx;
                        let b_idx_off = b_idx * heads * b_shape[2] * n + h * b_shape[2] * n;
                        let b_idx_val = b_idx_off + k_idx * n + j;
                        sum += a_data[a_idx] * b_data[b_idx_val];
                    }
                    let out_idx = b_idx * heads * m * n + h * m * n + i * n + j;
                    output[out_idx] = sum;
                }
            }
        }
    }

    Tensor::new(&output, &[batch, heads, m, n])
}

/// Scale tensor by a scalar
fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| v * scale).collect();
    Tensor::new(&data, x.shape())
}

/// Add mask to attention scores
fn add_mask(scores: &Tensor, mask: &Tensor) -> Tensor {
    let data: Vec<f32> =
        scores.data().iter().zip(mask.data().iter()).map(|(&s, &m)| s + m).collect();
    Tensor::new(&data, scores.shape())
}

/// Softmax over last dimension
fn softmax_last_dim(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let mut output = vec![0.0; x.data().len()];
    let x_data = x.data();

    for b in 0..batch_size {
        let offset = b * last_dim;
        let slice = &x_data[offset..offset + last_dim];

        let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = slice.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        for i in 0..last_dim {
            output[offset + i] = if sum > 0.0 { exp_vals[i] / sum } else { 0.0 };
        }
    }

    Tensor::new(&output, shape)
}

/// Create causal mask for attention
fn create_causal_mask(size: usize) -> Result<Tensor> {
    let mut data = vec![0.0; size * size];

    for i in 0..size {
        for j in 0..size {
            if j > i {
                data[i * size + j] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(Tensor::new(&data, &[size, size]))
}

/// Scaled dot-product attention
fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_layer: Option<&Dropout>,
) -> (Tensor, Tensor) {
    let d_k = query.shape()[query.shape().len() - 1] as f32;
    let scale = 1.0 / d_k.sqrt();

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    let key_t = transpose_last_two(key);
    let scores = matmul_batched_4d(query, &key_t);
    let scores = scale_tensor(&scores, scale);

    // Apply mask
    let scores = match attn_mask {
        Some(mask) => {
            // Mask shape is [seq_len, total_seq_len], scores shape is [batch, heads, seq_len, total_seq_len]
            // We need to broadcast the mask to match scores
            let scores_shape = scores.shape();
            let (batch, heads, q_len, k_len) = (
                scores_shape[0],
                scores_shape[1],
                scores_shape[2],
                scores_shape[3],
            );
            let mask_shape = mask.shape();
            let mask_q_len = mask_shape[0];
            let mask_k_len = mask_shape[1];

            // Create broadcasted mask: [batch, heads, q_len, k_len]
            let mut broadcasted_mask_data = vec![0.0; batch * heads * q_len * k_len];
            let mask_data = mask.data();

            for b in 0..batch {
                for h in 0..heads {
                    for q in 0..q_len {
                        for k in 0..k_len {
                            // Use mask value if within bounds, otherwise 0.0 (no masking)
                            let mask_val = if q < mask_q_len && k < mask_k_len {
                                mask_data[q * mask_k_len + k]
                            } else {
                                0.0
                            };
                            let idx = b * heads * q_len * k_len + h * q_len * k_len + q * k_len + k;
                            broadcasted_mask_data[idx] = mask_val;
                        }
                    }
                }
            }

            let broadcasted_mask =
                Tensor::new(&broadcasted_mask_data, &[batch, heads, q_len, k_len]);
            add_mask(&scores, &broadcasted_mask)
        }
        None => scores,
    };

    // Softmax over last dimension
    let attn_weights = softmax_last_dim(&scores);

    // Apply dropout using aprender's Dropout layer (if provided)
    let attn_weights = if let Some(dropout) = dropout_layer {
        dropout.forward(&attn_weights)
    } else {
        attn_weights
    };

    // Weighted sum: attn_weights @ V
    let output = matmul_batched_4d(&attn_weights, value);

    (output, attn_weights)
}

/// Slice RoPE embeddings to a specific sequence length
fn slice_rope(rope: &Tensor, seq_len: usize) -> Result<Tensor> {
    let shape = rope.shape();
    if shape.len() != 4 {
        anyhow::bail!("Expected 4D RoPE tensor, got shape {:?}", shape);
    }

    let max_seq_len = shape[1];
    let half_dim = shape[3];

    if seq_len > max_seq_len {
        anyhow::bail!(
            "Requested seq_len {} exceeds max_seq_len {}",
            seq_len,
            max_seq_len
        );
    }

    // Extract slice: [1, seq_len, 1, half_dim]
    let rope_data = rope.data();
    let slice_data: Vec<f32> = rope_data[..seq_len * half_dim].to_vec();
    Ok(Tensor::new(&slice_data, &[1, seq_len, 1, half_dim]))
}

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
/// This implements GQA with full KV cache support, RoPE, and QK normalization.
/// We use our own projections instead of wrapping GQA to have full control.
pub struct CausalSelfAttention {
    /// Query projection: n_embd -> n_embd
    q_proj: Linear,
    /// Key projection: n_embd -> n_kv_head * head_dim
    k_proj: Linear,
    /// Value projection: n_embd -> n_kv_head * head_dim
    v_proj: Linear,
    /// Output projection: n_embd -> n_embd
    out_proj: Linear,
    /// Number of query heads
    n_head: usize,
    /// Number of key/value heads
    n_kv_head: usize,
    /// Head dimension
    head_dim: usize,
    /// Embedding dimension
    n_embd: usize,
    /// Dropout layer (None if dropout_p == 0.0)
    dropout_layer: Option<Dropout>,
    /// Training mode
    training: bool,
}

impl CausalSelfAttention {
    /// Create a new CausalSelfAttention layer
    ///
    /// # Arguments
    /// * `n_embd` - Embedding dimension
    /// * `n_head` - Number of query heads
    /// * `n_kv_head` - Number of key/value heads (for GQA)
    /// * `dropout_p` - Dropout probability (0.0 = no dropout)
    /// * `seed` - Optional random seed for reproducibility (None = non-deterministic)
    pub fn new(
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        dropout_p: Option<f32>,
        seed: Option<u64>,
    ) -> Self {
        let head_dim = n_embd / n_head;
        let kv_dim = n_kv_head * head_dim;

        // Create dropout layer if needed (dropout_p will be set via with_dropout if needed)
        let dropout_layer = if let Some(dropout_p) = dropout_p {
            match (dropout_p, seed) {
                (dropout_p, Some(s)) if dropout_p > 0.0 => Some(Dropout::with_seed(dropout_p, s)),
                (dropout_p, None) if dropout_p > 0.0 => Some(Dropout::new(dropout_p)),
                _ => None,
            }
        } else {
            None
        };

        Self {
            q_proj: Linear::new(n_embd, n_embd),
            k_proj: Linear::new(n_embd, kv_dim),
            v_proj: Linear::new(n_embd, kv_dim),
            out_proj: Linear::new(n_embd, n_embd),
            n_head,
            n_kv_head,
            head_dim,
            n_embd,
            dropout_layer,
            training: true,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        if let Some(ref mut dropout) = self.dropout_layer {
            if training {
                dropout.train();
            } else {
                dropout.eval();
            }
        }
    }

    /// Forward pass with optional KV cache and causal masking
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    /// * `kv_cache` - Optional KV cache for inference (None during training)
    /// * `layer_idx` - Layer index for KV cache (used when kv_cache is Some)
    /// * `cos_sin` - Optional RoPE cos/sin tensors for positional encoding
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, n_embd]
    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
        cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            anyhow::bail!(
                "Expected 3D tensor [batch, seq_len, n_embd], got shape {:?}",
                shape
            );
        }

        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape Q: [batch, seq, embed] -> [batch, num_heads, seq, head_dim]
        let q = reshape_for_attention(&q, batch_size, seq_len, self.n_head, self.head_dim);

        // Reshape K, V: [batch, seq, kv_dim] -> [batch, num_kv_heads, seq, head_dim]
        let k = reshape_for_attention(&k, batch_size, seq_len, self.n_kv_head, self.head_dim);
        let v = reshape_for_attention(&v, batch_size, seq_len, self.n_kv_head, self.head_dim);

        // Handle KV cache if provided
        let (k_use, v_use, total_seq_len) = if let Some(cache) = kv_cache {
            // Insert new K/V into cache and get concatenated result
            let (k_cached, v_cached) = cache.insert_kv(layer_idx, k, v)?;
            let seq_len_total = k_cached.shape()[2];
            (k_cached, v_cached, seq_len_total)
        } else {
            // No cache - use current K/V directly
            (k, v, seq_len)
        };

        // Expand K, V to match Q heads by repeating (for GQA)
        let groups = self.n_head / self.n_kv_head;
        let k_expanded = repeat_kv_heads(&k_use, groups);
        let v_expanded = repeat_kv_heads(&v_use, groups);

        // Apply RoPE to Q and K if provided
        let (q_rope, k_rope) = if let Some((cos, sin)) = cos_sin {
            // Slice RoPE embeddings to current sequence length
            // cos/sin are [1, max_seq_len, 1, head_dim/2]
            // We need [1, seq_len, 1, head_dim/2] for Q and [1, total_seq_len, 1, head_dim/2] for K
            let cos_q = slice_rope(cos, seq_len)?;
            let sin_q = slice_rope(sin, seq_len)?;
            let cos_k = slice_rope(cos, total_seq_len)?;
            let sin_k = slice_rope(sin, total_seq_len)?;

            // Apply RoPE to Q and K
            let q_rope = crate::rope::apply_rotary_emb(&q, &cos_q, &sin_q)?;
            let k_rope = crate::rope::apply_rotary_emb(&k_expanded, &cos_k, &sin_k)?;
            (q_rope, k_rope)
        } else {
            // No RoPE - use Q and K directly
            (q, k_expanded)
        };

        // Apply QK normalization
        let (q_norm, k_norm) = apply_qk_norm(&q_rope, &k_rope)?;

        // Create causal mask for attention
        // Mask shape needs to match scores: [batch, heads, seq_len, total_seq_len]
        // But create_causal_mask creates [seq_len, total_seq_len], so we'll handle broadcasting
        let causal_mask = if total_seq_len > 1 && seq_len > 1 {
            // Create mask for current query positions attending to all key positions
            // Shape: [seq_len, total_seq_len]
            Some(create_causal_mask(total_seq_len)?)
        } else {
            None
        };

        // Compute scaled dot-product attention
        // Pass dropout layer if training and dropout is enabled
        let dropout_ref = if self.training {
            self.dropout_layer.as_ref()
        } else {
            None
        };
        let (attn_output, _attn_weights) = scaled_dot_product_attention(
            &q_norm,
            &k_norm,
            &v_expanded,
            causal_mask.as_ref(),
            dropout_ref,
        );

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let attn_output = reshape_from_attention(&attn_output, batch_size, seq_len, self.n_embd);

        // Output projection
        let output = self.out_proj.forward(&attn_output);

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

impl Module for CausalSelfAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Use forward without KV cache for Module trait compatibility
        self.forward(input, None, 0, None).expect("Attention forward failed")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        // Dropout has no learnable parameters
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters_mut());
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        // Dropout has no learnable parameters
        params
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
        Self { cache: Vec::new() }
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
    pub fn insert_kv(
        &mut self,
        layer_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Ensure cache has enough layers
        while self.cache.len() <= layer_idx {
            // Create empty tensors for this layer
            // Shape will be determined by first insert
            self.cache.push((Tensor::zeros(&[0]), Tensor::zeros(&[0])));
        }

        let (cached_k, cached_v) = &self.cache[layer_idx];

        // If cache is empty, just store the new k, v
        if cached_k.shape().is_empty() || cached_k.shape()[0] == 0 {
            self.cache[layer_idx] = (k.clone(), v.clone());
            return Ok((k, v));
        }

        // Concatenate along sequence dimension
        // K/V shapes: [batch, n_kv_heads, seq_len, head_dim]
        let k_shape = cached_k.shape();
        let v_shape = cached_v.shape();
        let new_k_shape = k.shape();

        // Verify shapes are compatible (batch, heads, head_dim must match)
        if k_shape.len() != 4 || new_k_shape.len() != 4 {
            anyhow::bail!(
                "Expected 4D tensors for K/V cache, got shapes {:?} and {:?}",
                k_shape,
                new_k_shape
            );
        }

        if k_shape[0] != new_k_shape[0]
            || k_shape[1] != new_k_shape[1]
            || k_shape[3] != new_k_shape[3]
        {
            anyhow::bail!(
                "Shape mismatch: cached {:?} vs new {:?} (batch, heads, head_dim must match)",
                k_shape,
                new_k_shape
            );
        }

        // Concatenate data along sequence dimension (dim 2)
        let new_seq_len = k_shape[2] + new_k_shape[2];

        // Concatenate k data
        let mut k_data = cached_k.data().to_vec();
        k_data.extend_from_slice(k.data());
        let k_concat = Tensor::new(&k_data, &[k_shape[0], k_shape[1], new_seq_len, k_shape[3]]);

        // Concatenate v data
        let mut v_data = cached_v.data().to_vec();
        v_data.extend_from_slice(v.data());
        let v_concat = Tensor::new(&v_data, &[v_shape[0], v_shape[1], new_seq_len, v_shape[3]]);

        self.cache[layer_idx] = (k_concat.clone(), v_concat.clone());
        Ok((k_concat, v_concat))
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
        let k = Tensor::ones(&[1, 2, 3, 4]); // 2 key/value heads

        let (q_norm, k_norm) = apply_qk_norm(&q, &k).unwrap();

        assert_eq!(q_norm.shape(), q.shape());
        assert_eq!(k_norm.shape(), k.shape());
    }

    #[test]
    fn test_causal_attention_creation() {
        let attn = CausalSelfAttention::new(768, 6, 6, None, None);
        assert_eq!(attn.n_head(), 6);
        assert_eq!(attn.n_kv_head(), 6);
        assert_eq!(attn.head_dim(), 128);
    }

    #[test]
    fn test_causal_attention_gqa() {
        // Test GQA with n_kv_head < n_head
        let attn = CausalSelfAttention::new(768, 6, 2, None, None);
        assert_eq!(attn.n_head(), 6);
        assert_eq!(attn.n_kv_head(), 2);
    }

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new();
        assert_eq!(cache.cache.len(), 0);
    }
}
