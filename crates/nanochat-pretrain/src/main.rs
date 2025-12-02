//! Pretraining stage for nanochat base language modeling
//!
//! This binary trains the base language model on raw text data.
//!
//! # Tokenizer Usage
//!
//! When implementing, use `nanochat_tokenizer::Tokenizer` which wraps `aprender::text::tokenize::BpeTokenizer`.
//! Ensure vocabulary size matches model config using `GPTConfig::with_tokenizer_vocab_size()` or
//! `validate_tokenizer_model_compatibility()` from `nanochat_model`.

fn main() {
    todo!("nanochat-pretrain: Pretraining stage (not yet implemented)");
}
