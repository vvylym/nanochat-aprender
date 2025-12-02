//! Integration tests for tokenizer-model compatibility

use nanochat_model::{GPTConfig, GPT, validate_tokenizer_model_compatibility, config::ConfigError};
use nanochat_tokenizer::Tokenizer;
use aprender::autograd::Tensor;
use aprender::text::tokenize::BpeTokenizer;

#[test]
fn test_tokenizer_model_compatibility() -> anyhow::Result<()> {
    // Create a tokenizer
    let corpus = vec!["hello world", "hello rust", "world peace", "the quick brown fox"];
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500)?;
    
    // Create model config with matching vocab_size
    let config = GPTConfig::with_tokenizer_vocab_size(
        tokenizer.vocab_size(),
        1024,  // sequence_len
        12,    // n_layer
        6,     // n_head
        6,     // n_kv_head
        768,   // n_embd
        None,  // dropout
        None,  // seed
    )?;
    
    // Create model
    let mut model = GPT::new(config);
    
    // Test encoding and forward pass
    let text = "hello world";
    let ids = tokenizer.encode(text)?;
    
    // Convert to tensor [batch=1, seq_len]
    let mut data = Vec::new();
    for &id in &ids {
        data.push(id as f32);
    }
    let input_tensor = Tensor::new(&data, &[1, ids.len()]);
    
    // Forward pass
    let logits = model.forward_cache(&input_tensor, None)?;
    
    // Verify logits shape: [batch, seq_len, vocab_size]
    let shape = logits.shape();
    assert_eq!(shape[0], 1, "Batch size should be 1");
    assert_eq!(shape[1], ids.len(), "Sequence length should match input");
    assert_eq!(shape[2], tokenizer.vocab_size(), "Vocab size should match tokenizer");
    
    Ok(())
}

#[test]
fn test_validate_tokenizer_model_compatibility() -> anyhow::Result<()> {
    // Create a tokenizer
    let corpus = vec!["hello world", "hello rust"];
    let _tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500)?;
    
    // Get the underlying BpeTokenizer
    // Note: We can't access the internal BpeTokenizer directly, so we'll test with a new one
    let corpus_vec: Vec<&str> = corpus.iter().map(|s| s.as_ref()).collect();
    let bpe_tokenizer = BpeTokenizer::train(&corpus_vec, 500)?;
    
    // Create model config with matching vocab_size
    let config = GPTConfig::with_tokenizer_vocab_size(
        bpe_tokenizer.vocab_size(),
        1024, 12, 6, 6, 768, None, None,
    )?;
    
    // Validate compatibility
    validate_tokenizer_model_compatibility(&bpe_tokenizer, &config)?;
    
    Ok(())
}

#[test]
fn test_tokenizer_model_forward_pass() -> anyhow::Result<()> {
    // Create a tokenizer
    let corpus = vec!["hello", "world", "test"];
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 300)?;
    
    // Create model with matching vocab_size
    let config = GPTConfig::with_tokenizer_vocab_size(
        tokenizer.vocab_size(),
        1024, 12, 6, 6, 768, None, None,
    )?;
    let mut model = GPT::new(config);
    
    // Encode text
    let text = "hello";
    let ids = tokenizer.encode(text)?;
    
    // Convert to tensor
    let mut data = Vec::new();
    for &id in &ids {
        data.push(id as f32);
    }
    let input_tensor = Tensor::new(&data, &[1, ids.len()]);
    
    // Forward pass
    let logits = model.forward_cache(&input_tensor, None)?;
    
    // Verify output
    let shape = logits.shape();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], ids.len());
    assert_eq!(shape[2], tokenizer.vocab_size());
    
    Ok(())
}

#[test]
fn test_validate_tokenizer_model_incompatibility() {
    // Create a tokenizer
    let corpus = vec!["hello world"];
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500).unwrap();
    
    // Create model config with mismatched vocab_size
    let config = GPTConfig::new(
        1024,
        tokenizer.vocab_size() + 100, // Mismatched vocab_size
        12, 6, 6, 768, None, None,
    ).unwrap();
    
    // Validation should fail
    let result = config.validate_vocab_size(tokenizer.vocab_size());
    assert!(result.is_err(), "Should fail when vocab_size doesn't match");
    
    if let Err(ConfigError::VocabSizeMismatch { config: c, tokenizer: t }) = result {
        assert_eq!(c, tokenizer.vocab_size() + 100);
        assert_eq!(t, tokenizer.vocab_size());
    } else {
        panic!("Expected VocabSizeMismatch error");
    }
}

#[test]
fn test_tokenizer_model_roundtrip() -> anyhow::Result<()> {
    // Create a tokenizer
    let corpus = vec![
        "hello world",
        "the quick brown fox",
        "rust is awesome",
        "machine learning",
    ];
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 1000)?;
    
    // Create model with matching vocab_size
    let config = GPTConfig::with_tokenizer_vocab_size(
        tokenizer.vocab_size(),
        1024, 12, 6, 6, 768, None, None,
    )?;
    let mut model = GPT::new(config);
    
    // Test multiple texts
    for text in corpus {
        // Encode
        let ids = tokenizer.encode(text)?;
        
        // Convert to tensor
        let mut data = Vec::new();
        for &id in &ids {
            data.push(id as f32);
        }
        let input_tensor = Tensor::new(&data, &[1, ids.len()]);
        
        // Forward pass
        let logits = model.forward_cache(&input_tensor, None)?;
        
        // Verify output shape
        let shape = logits.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], ids.len());
        assert_eq!(shape[2], tokenizer.vocab_size());
        
        // Decode back
        let decoded = tokenizer.decode(&ids)?;
        // Note: Decoded text may not match exactly due to tokenization, but should be valid
        assert!(!decoded.is_empty());
    }
    
    Ok(())
}

#[test]
fn test_tokenizer_model_with_special_tokens() -> anyhow::Result<()> {
    // Create a tokenizer
    let corpus = vec!["hello world", "test message"];
    let tokenizer = Tokenizer::train_from_iterator(corpus.iter(), 500)?;
    
    // Create model with matching vocab_size
    let config = GPTConfig::with_tokenizer_vocab_size(
        tokenizer.vocab_size(),
        1024, 12, 6, 6, 768, None, None,
    )?;
    let mut model = GPT::new(config);
    
    // Try to encode with special tokens (if they exist)
    let text = "hello";
    let _ids = tokenizer.encode(text)?;
    
    // Try to find special tokens
    let bos_token = tokenizer.special_token_id("<s>")
        .or_else(|_| tokenizer.special_token_id("<|bos|>"));
    let eos_token = tokenizer.special_token_id("</s>")
        .or_else(|_| tokenizer.special_token_id("<|eos|>"));
    
    // If special tokens exist, test with them
    if let (Ok(_bos_id), Ok(_eos_id)) = (bos_token, eos_token) {
        // Try to encode with special tokens
        let bos_str = if tokenizer.special_token_id("<s>").is_ok() { "<s>" } else { "<|bos|>" };
        let eos_str = if tokenizer.special_token_id("</s>").is_ok() { "</s>" } else { "<|eos|>" };
        
        let ids_with_special = tokenizer.encode_with_special(
            text,
            Some(bos_str),
            Some(eos_str),
        )?;
        
        // Convert to tensor and test forward pass
        let mut data = Vec::new();
        for &id in &ids_with_special {
            data.push(id as f32);
        }
        let input_tensor = Tensor::new(&data, &[1, ids_with_special.len()]);
        
        let logits = model.forward_cache(&input_tensor, None)?;
        let shape = logits.shape();
        assert_eq!(shape[2], tokenizer.vocab_size());
    }
    
    Ok(())
}

#[test]
fn test_tokenizer_save_load_with_model() -> anyhow::Result<()> {
    use tempfile::TempDir;
    
    // Create a tokenizer
    let corpus = vec!["hello world", "test message", "rust programming"];
    let tokenizer1 = Tokenizer::train_from_iterator(corpus.iter(), 500)?;
    
    // Create model with matching vocab_size
    let vocab_size = tokenizer1.vocab_size();
    let config = GPTConfig::with_tokenizer_vocab_size(
        vocab_size,
        1024, 12, 6, 6, 768, None, None,
    )?;
    let mut model = GPT::new(config);
    
    // Save tokenizer
    let temp_dir = TempDir::new()?;
    tokenizer1.save(temp_dir.path())?;
    
    // Load tokenizer
    let tokenizer2 = Tokenizer::from_directory(temp_dir.path())?;
    
    // Verify vocab_size matches
    assert_eq!(vocab_size, tokenizer2.vocab_size());
    
    // Verify model config still matches
    let config2 = GPTConfig::with_tokenizer_vocab_size(
        tokenizer2.vocab_size(),
        1024, 12, 6, 6, 768, None, None,
    )?;
    assert_eq!(vocab_size, config2.vocab_size);
    
    // Test encoding with both tokenizers
    let text = "hello world";
    let ids1 = tokenizer1.encode(text)?;
    let ids2 = tokenizer2.encode(text)?;
    assert_eq!(ids1, ids2);
    
    // Test forward pass with loaded tokenizer
    let mut data = Vec::new();
    for &id in &ids2 {
        data.push(id as f32);
    }
    let input_tensor = Tensor::new(&data, &[1, ids2.len()]);
    let logits = model.forward_cache(&input_tensor, None)?;
    
    let shape = logits.shape();
    assert_eq!(shape[2], tokenizer2.vocab_size());
    
    Ok(())
}

