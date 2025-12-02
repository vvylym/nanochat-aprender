//! Supervised fine-tuning stage for nanochat instruction following
//!
//! This binary fine-tunes the model for instruction-following tasks.
//!
//! # Tokenizer Usage
//!
//! When implementing, use `nanochat_tokenizer::Tokenizer` which wraps `aprender::text::tokenize::BpeTokenizer`.
//! Ensure vocabulary size matches model config using `GPTConfig::with_tokenizer_vocab_size()` or
//! `validate_tokenizer_model_compatibility()` from `nanochat_model`.

fn main() {
    todo!("nanochat-sft: Supervised fine-tuning stage (not yet implemented)");
}
