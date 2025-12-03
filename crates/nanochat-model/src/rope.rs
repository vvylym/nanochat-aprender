//! Rotary Position Embeddings (RoPE)

use anyhow::Result;
use aprender::autograd::Tensor;

/// Precompute rotary position embeddings for a given sequence length and head dimension
///
/// This precomputes cos and sin frequencies for efficient application during forward pass.
/// The frequencies are computed based on the base theta (default 10000).
///
/// # Arguments
/// * `seq_len` - Maximum sequence length to precompute
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `base` - Base frequency for RoPE (default: 10000.0)
///
/// # Returns
/// Tuple of (cos, sin) tensors with shape [1, seq_len, 1, head_dim/2] for broadcasting
pub fn precompute_rotary_embeddings(
    seq_len: usize,
    head_dim: usize,
    base: f32,
) -> Result<(Tensor, Tensor)> {
    if !head_dim.is_multiple_of(2) {
        anyhow::bail!("head_dim must be even for RoPE, got {}", head_dim);
    }

    let half_dim = head_dim / 2;

    // Compute inverse frequencies: 1 / (base^(2i/d))
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    // Compute cos and sin for each position
    let mut cos_data = Vec::with_capacity(seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(seq_len * half_dim);

    for pos in 0..seq_len {
        for &freq in &inv_freq {
            let angle = pos as f32 * freq;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }

    // Create tensors with shape [1, seq_len, 1, half_dim] for broadcasting
    // This matches the Python implementation: cos[None, :, None, :]
    let cos = Tensor::new(&cos_data, &[1, seq_len, 1, half_dim]);
    let sin = Tensor::new(&sin_data, &[1, seq_len, 1, half_dim]);

    Ok((cos, sin))
}

/// Apply rotary position embeddings to a tensor
///
/// This applies RoPE to queries or keys for relative positional encoding.
/// The input tensor should have shape [batch, n_heads, seq_len, head_dim].
///
/// Implementation matches the Python version:
/// - Split last dimension into two halves
/// - Rotate pairs: y1 = x1 * cos + x2 * sin, y2 = x1 * (-sin) + x2 * cos
/// - Re-assemble
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, n_heads, seq_len, head_dim]
/// * `cos` - Precomputed cosine frequencies [1, seq_len, 1, head_dim/2]
/// * `sin` - Precomputed sine frequencies [1, seq_len, 1, head_dim/2]
///
/// # Returns
/// Tensor with rotary embeddings applied, same shape as input
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let shape = x.shape();
    if shape.len() != 4 {
        anyhow::bail!(
            "Expected 4D tensor [batch, n_heads, seq_len, head_dim], got shape {:?}",
            shape
        );
    }

    let head_dim = shape[3];
    if !head_dim.is_multiple_of(2) {
        anyhow::bail!("head_dim must be even, got {}", head_dim);
    }

    let half_dim = head_dim / 2;
    let batch = shape[0];
    let n_heads = shape[1];
    let seq_len = shape[2];

    // Verify cos/sin shapes match
    let cos_shape = cos.shape();
    let sin_shape = sin.shape();
    if cos_shape.len() != 4 || sin_shape.len() != 4 {
        anyhow::bail!("cos and sin must be 4D tensors");
    }
    if cos_shape[1] != seq_len || sin_shape[1] != seq_len {
        anyhow::bail!(
            "cos/sin sequence length {} doesn't match input sequence length {}",
            cos_shape[1],
            seq_len
        );
    }
    if cos_shape[3] != half_dim || sin_shape[3] != half_dim {
        anyhow::bail!(
            "cos/sin half_dim {} doesn't match expected {}",
            cos_shape[3],
            half_dim
        );
    }

    // Get input data
    let x_data = x.data();
    let cos_data = cos.data();
    let sin_data = sin.data();

    // Apply RoPE: split into halves, rotate, re-assemble
    let mut output = vec![0.0; x_data.len()];

    for b in 0..batch {
        for h in 0..n_heads {
            for s in 0..seq_len {
                let x_offset =
                    (b * n_heads * seq_len * head_dim) + (h * seq_len * head_dim) + (s * head_dim);
                let cos_offset = s * half_dim; // cos/sin are [1, seq_len, 1, half_dim]
                let out_offset = x_offset;

                // Split into two halves
                let x1 = &x_data[x_offset..x_offset + half_dim];
                let x2 = &x_data[x_offset + half_dim..x_offset + head_dim];
                let cos_vals = &cos_data[cos_offset..cos_offset + half_dim];
                let sin_vals = &sin_data[cos_offset..cos_offset + half_dim];

                // Rotate: y1 = x1 * cos + x2 * sin, y2 = x1 * (-sin) + x2 * cos
                for i in 0..half_dim {
                    output[out_offset + i] = x1[i] * cos_vals[i] + x2[i] * sin_vals[i];
                    output[out_offset + half_dim + i] =
                        x1[i] * (-sin_vals[i]) + x2[i] * cos_vals[i];
                }
            }
        }
    }

    Ok(Tensor::new(&output, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_precompute() {
        let (cos, sin) =
            precompute_rotary_embeddings(10, 64, 10000.0).expect("Failed to precompute RoPE");

        assert_eq!(cos.shape(), &[1, 10, 1, 32]);
        assert_eq!(sin.shape(), &[1, 10, 1, 32]);
    }

    #[test]
    fn test_rope_precompute_odd_head_dim() {
        let result = precompute_rotary_embeddings(10, 65, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_apply() {
        // Create input tensor [batch=1, n_heads=2, seq_len=3, head_dim=4]
        let x = Tensor::ones(&[1, 2, 3, 4]);
        let (cos, sin) =
            precompute_rotary_embeddings(3, 4, 10000.0).expect("Failed to precompute RoPE");

        let result = apply_rotary_emb(&x, &cos, &sin).expect("RoPE application failed");

        assert_eq!(result.shape(), x.shape());
    }
}
