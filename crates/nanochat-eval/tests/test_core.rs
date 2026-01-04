//! Unit tests for CORE benchmark

use anyhow::Result;
use nanochat_eval::core::evaluate_core;
use nanochat_model::{GPTConfig, GPT};
use nanochat_tokenizer::Tokenizer;
use tempfile::TempDir;

/// Create a minimal test model
fn create_test_model() -> GPT {
    let config = GPTConfig {
        vocab_size: 1000,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 128,
        sequence_len: 256,
        dropout: Some(0.0),
        seed: Some(42),
    };
    GPT::new(config)
}

/// Create a minimal test tokenizer
fn create_test_tokenizer() -> Tokenizer {
    let corpus = ["hello world", "test data", "evaluation", "benchmark"];
    Tokenizer::train_from_iterator(corpus.iter(), 1000).expect("Failed to create test tokenizer")
}

/// Create a temporary directory with CORE benchmark data
fn create_test_data_dir() -> Result<TempDir> {
    use std::fs;
    use std::io::Write;

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let data_dir = temp_dir.path();

    // Create a minimal CORE benchmark data file
    // Format: JSONL with prompts and expected completions
    let jsonl_content = r#"{"prompt": "The capital of France is", "expected": "Paris"}
{"prompt": "Two plus two equals", "expected": "four"}
"#;

    let jsonl_path = data_dir.join("core.jsonl");
    let mut file = fs::File::create(&jsonl_path).expect("Failed to create test file");
    file.write_all(jsonl_content.as_bytes()).expect("Failed to write test data");
    file.flush().expect("Failed to flush");

    Ok(temp_dir)
}

#[test]
fn test_core_benchmark_structure() -> Result<()> {
    // Test that we can parse CORE benchmark data format
    let jsonl_line = r#"{"prompt": "The capital of France is", "expected": "Paris"}"#;

    let parsed: serde_json::Value = serde_json::from_str(jsonl_line).expect("Failed to parse JSON");

    assert!(parsed.get("prompt").is_some());
    assert!(parsed.get("expected").is_some());
    assert_eq!(
        parsed
            .get("prompt")
            .expect("prompt field should exist")
            .as_str()
            .expect("prompt should be a string"),
        "The capital of France is"
    );

    Ok(())
}

#[test]
fn test_core_evaluation_function_exists() -> Result<()> {
    // Test that the evaluate_core function exists and has correct signature
    let temp_dir = create_test_data_dir().expect("Failed to create test data");
    let data_dir = temp_dir.path();

    let model = create_test_model();
    let tokenizer = create_test_tokenizer();

    // Verify the function can be called
    let result = evaluate_core(&model, &tokenizer, data_dir, 1, 64)
        .expect("evaluate_core should be callable");

    // Verify result structure
    assert_eq!(result.benchmark_name, "CORE");
    assert!(result.score >= 0.0 && result.score <= 1.0);

    Ok(())
}
