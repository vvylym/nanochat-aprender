//! Command-line interface for nanochat with real-time progress
//!
//! This binary provides CLI commands for training, evaluation, and inference.
//!
//! # Tokenizer Usage
//!
//! When implementing, use `nanochat_tokenizer::Tokenizer` which wraps `aprender::text::tokenize::BpeTokenizer`.
//! Ensure vocabulary size matches model config using `GPTConfig::with_tokenizer_vocab_size()` or
//! `validate_tokenizer_model_compatibility()` from `nanochat_model`.
//! Consider adding a `--validate-vocab-size` flag to validate tokenizer-model compatibility.

fn main() {
    todo!("nanochat-cli: CLI interface (not yet implemented)");
}
