//! Unit tests for conversational data loading in mid-training

use anyhow::Result;
use nanochat_midtrain::dataloader::ConversationDataLoader;
use nanochat_tokenizer::Tokenizer;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

/// Create a temporary directory with conversational data files
fn create_test_data_dir() -> Result<TempDir> {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let data_dir = temp_dir.path();

    // Create a JSONL file with conversational data
    let jsonl_content = r#"{"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]}
{"messages": [{"role": "user", "content": "What is Rust?"}, {"role": "assistant", "content": "Rust is a systems programming language."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Write a haiku"}, {"role": "assistant", "content": "Code flows like water,\nMemory safe and fast,\nRust brings joy to all."}]}
"#;

    let jsonl_path = data_dir.join("conversations.jsonl");
    let mut file = fs::File::create(&jsonl_path).expect("Failed to create test file");
    file.write_all(jsonl_content.as_bytes()).expect("Failed to write test data");
    file.flush().expect("Failed to flush");

    Ok(temp_dir)
}

/// Create a minimal tokenizer for testing
fn create_test_tokenizer() -> Tokenizer {
    let corpus = ["hello world", "test data", "conversation"];
    Tokenizer::train_from_iterator(corpus.iter(), 100).expect("Failed to create test tokenizer")
}

#[test]
fn test_conversation_dataloader_creation() -> Result<()> {
    let temp_dir = create_test_data_dir().expect("Failed to create test data");
    let data_dir = temp_dir.path();

    let tokenizer = create_test_tokenizer();

    let dataloader = ConversationDataLoader::new(
        data_dir,
        tokenizer,
        2,        // batch_size
        128,      // seq_len
        1,        // num_workers
        Some(42), // seed
    )?;

    // Verify dataloader was created successfully
    assert_eq!(dataloader.batch_size(), 2);
    assert_eq!(dataloader.seq_len(), 128);

    Ok(())
}

#[test]
fn test_load_conversational_data() -> Result<()> {
    let temp_dir = create_test_data_dir().expect("Failed to create test data");
    let data_dir = temp_dir.path();

    let tokenizer = create_test_tokenizer();

    let dataloader = ConversationDataLoader::new(
        data_dir,
        tokenizer,
        1,        // batch_size
        64,       // seq_len
        1,        // num_workers
        Some(42), // seed
    )?;

    // Verify that conversations were loaded
    assert!(dataloader.conversation_count() > 0);

    Ok(())
}

#[test]
fn test_conversation_format_parsing() -> Result<()> {
    // Test that we can parse the JSONL conversation format correctly
    let jsonl_line = r#"{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}"#;

    // Parse the JSON
    let parsed: serde_json::Value = serde_json::from_str(jsonl_line).expect("Failed to parse JSON");

    // Verify structure
    assert!(parsed.get("messages").is_some());
    let messages = parsed
        .get("messages")
        .expect("messages field should exist")
        .as_array()
        .expect("messages should be an array");

    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0]
            .get("role")
            .expect("role should exist")
            .as_str()
            .expect("role should be a string"),
        "user"
    );
    assert_eq!(
        messages[1]
            .get("role")
            .expect("role should exist")
            .as_str()
            .expect("role should be a string"),
        "assistant"
    );

    Ok(())
}

#[test]
fn test_conversation_to_sequence() -> Result<()> {
    // Test converting a conversation to a token sequence
    let temp_dir = create_test_data_dir().expect("Failed to create test data");
    let data_dir = temp_dir.path();

    let tokenizer = create_test_tokenizer();

    let mut dataloader = ConversationDataLoader::new(data_dir, tokenizer, 1, 128, 1, Some(42))?;

    // Get a batch and verify it contains tokenized conversation data
    let batch_opt = dataloader.next_batch().expect("Should be able to get a batch");
    if let Some((inputs, _targets)) = batch_opt {
        assert_eq!(inputs.shape()[0], 1); // batch size
        assert_eq!(inputs.shape()[1], 128); // sequence length
    }

    Ok(())
}

#[test]
fn test_dataloader_state_serialization() -> Result<()> {
    // Test that DataLoaderState can be serialized/deserialized for checkpointing
    use nanochat_midtrain::dataloader::DataLoaderState;

    let state = DataLoaderState {
        current_pos: 100,
        rng_seed: 42,
    };

    let serialized = serde_json::to_string(&state).expect("Should serialize");
    let deserialized: DataLoaderState =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.current_pos, 100);
    assert_eq!(deserialized.rng_seed, 42);

    Ok(())
}
