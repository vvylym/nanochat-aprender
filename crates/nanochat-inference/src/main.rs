//! OpenAI-compatible inference server for nanochat
//!
//! This binary provides an HTTP server with OpenAI Chat Completions API compatibility.
//!
//! # Tokenizer Usage
//!
//! When implementing, use `nanochat_tokenizer::Tokenizer` which wraps `aprender::text::tokenize::BpeTokenizer`.
//! Ensure vocabulary size matches model config using `GPTConfig::with_tokenizer_vocab_size()` or
//! `validate_tokenizer_model_compatibility()` from `nanochat_model`.

fn main() {
    todo!("nanochat-inference: Inference server (not yet implemented)");
}
