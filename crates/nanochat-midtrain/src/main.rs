//! Mid-training stage for nanochat conversational fine-tuning
//!
//! This binary fine-tunes the pretrained model for conversational tasks.
//!
//! # Tokenizer Usage
//!
//! When implementing, use `nanochat_tokenizer::Tokenizer` which wraps `aprender::text::tokenize::BpeTokenizer`.
//! Ensure vocabulary size matches model config using `GPTConfig::with_tokenizer_vocab_size()` or
//! `validate_tokenizer_model_compatibility()` from `nanochat_model`.

fn main() {
    todo!("nanochat-midtrain: Mid-training stage (not yet implemented)");
}
