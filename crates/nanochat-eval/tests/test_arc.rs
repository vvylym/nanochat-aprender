//! Unit tests for ARC benchmark

use anyhow::Result;
use nanochat_eval::arc::evaluate_arc;
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

/// Create a temporary directory with ARC benchmark data
fn create_test_data_dir() -> Result<TempDir> {
    use std::fs;
    use std::io::Write;

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let data_dir = temp_dir.path();

    // Create a minimal ARC benchmark data file
    let jsonl_content = r#"{"question": "What is the capital of France?", "choices": ["Paris", "London", "Berlin"], "answer": "Paris"}
{"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer": "4"}
"#;

    let jsonl_path = data_dir.join("arc_easy.jsonl");
    let mut file = fs::File::create(&jsonl_path).expect("Failed to create test file");
    file.write_all(jsonl_content.as_bytes()).expect("Failed to write test data");
    file.flush().expect("Failed to flush");

    Ok(temp_dir)
}

#[test]
fn test_arc_benchmark_structure() -> Result<()> {
    // Test that we can parse ARC benchmark data format
    let jsonl_line = r#"{"question": "What is the capital of France?", "choices": ["Paris", "London"], "answer": "Paris"}"#;

    let parsed: serde_json::Value = serde_json::from_str(jsonl_line).expect("Failed to parse JSON");

    assert!(parsed.get("question").is_some());
    assert!(parsed.get("choices").is_some());
    assert!(parsed.get("answer").is_some());

    Ok(())
}

#[test]
fn test_arc_evaluation_function_exists() -> Result<()> {
    let temp_dir = create_test_data_dir().expect("Failed to create test data");
    let data_dir = temp_dir.path();

    let model = create_test_model();
    let tokenizer = create_test_tokenizer();

    let result =
        evaluate_arc(&model, &tokenizer, data_dir, 1, 64).expect("evaluate_arc should be callable");

    assert_eq!(result.benchmark_name, "ARC");
    assert!(result.score >= 0.0 && result.score <= 1.0);

    Ok(())
}
