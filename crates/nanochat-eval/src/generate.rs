//! Text generation utilities for evaluation

use anyhow::Result;
use aprender::autograd::Tensor;
use nanochat_model::{attention::KVCache, GPT};
use nanochat_tokenizer::Tokenizer;

/// Generate text from a prompt using greedy decoding
///
/// # Arguments
/// * `model` - The GPT model
/// * `tokenizer` - The tokenizer
/// * `prompt` - Input prompt text
/// * `max_tokens` - Maximum tokens to generate
///
/// # Returns
/// Generated text (including the prompt)
pub fn generate_greedy(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    // Encode prompt
    let prompt_ids = tokenizer.encode(prompt)?;

    if prompt_ids.is_empty() {
        return Ok(prompt.to_string());
    }

    // Create initial input tensor [batch=1, seq_len] for the full prompt
    let mut input_ids = prompt_ids.clone();
    let mut kv_cache = KVCache::new();

    // First forward pass with the full prompt
    if !input_ids.is_empty() {
        let prompt_data: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
        let prompt_tensor = Tensor::new(&prompt_data, &[1, input_ids.len()]);
        let _logits = model.forward_cache(&prompt_tensor, Some(&mut kv_cache))?;
    }

    // Generate tokens one at a time
    for _ in 0..max_tokens {
        // Get current sequence length
        let current_len = input_ids.len();

        // Create tensor for last token only [batch=1, seq_len=1]
        let last_token = input_ids[current_len - 1];
        let input_tensor = Tensor::new(&[last_token as f32], &[1, 1]);

        // Forward pass with KV cache
        let logits = model.forward_cache(&input_tensor, Some(&mut kv_cache))?;

        // Get logits for last position [batch=1, vocab_size]
        let logits_shape = logits.shape();
        if logits_shape.len() != 3 || logits_shape[0] != 1 || logits_shape[1] != 1 {
            anyhow::bail!("Unexpected logits shape: {:?}", logits_shape);
        }

        // Extract logits for last token: [vocab_size]
        // For now, use a simple approach: get the last token's logits
        // This is a simplified version - in production, we'd use proper tensor slicing
        let vocab_size = logits_shape[2];

        // Greedy decoding: take argmax
        // For simplicity, we'll use a basic approach
        // In a full implementation, we'd use aprender's argmax operation
        let next_token = greedy_sample(&logits, vocab_size)?;

        // Check for end-of-sequence (using tokenizer's EOS token if available)
        // For now, we'll just append the token
        input_ids.push(next_token);

        // Simple stopping condition: if we've generated enough tokens
        if input_ids.len() >= prompt_ids.len() + max_tokens {
            break;
        }
    }

    // Decode the full sequence
    let generated_text = tokenizer.decode(&input_ids)?;
    Ok(generated_text)
}

/// Greedy sampling: take the token with highest logit
///
/// This is a simplified implementation. In production, we'd use aprender's
/// tensor operations for proper argmax.
fn greedy_sample(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    // Extract logits data
    // Logits shape is [batch=1, seq_len=1, vocab_size]
    // Data is laid out as [vocab_size] values for the last position
    let logits_data = logits.data();

    // For shape [1, 1, vocab_size], the data is just [vocab_size] values
    // Find argmax over the vocabulary dimension
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_idx = 0;

    // Take only the vocab_size elements (in case data is larger)
    for (idx, &logit) in logits_data.iter().enumerate().take(vocab_size) {
        if logit > max_logit {
            max_logit = logit;
            max_idx = idx;
        }
    }

    Ok(max_idx as u32)
}
