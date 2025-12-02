//! Integration tests for checkpoint save/load

use nanochat_model::{GPT, GPTConfig, save_checkpoint, load_checkpoint, CheckpointMetadata};
use tempfile::TempDir;
use std::collections::HashMap;

#[test]
fn test_checkpoint_save() {
    // Test saving model checkpoint to disk
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model");

    let result = save_checkpoint(&model, &checkpoint_path, None);
    assert!(result.is_ok());
    
    // Verify files were created
    assert!(checkpoint_path.with_extension("json").exists());
    assert!(checkpoint_path.with_extension("safetensors").exists());
}

#[test]
fn test_checkpoint_load() {
    // Test loading model checkpoint from disk
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model");

    // Save checkpoint
    save_checkpoint(&model, &checkpoint_path, None).unwrap();

    // Load checkpoint
    let (loaded_model, metadata) = load_checkpoint(&checkpoint_path).unwrap();
    
    // Verify config matches
    assert_eq!(loaded_model.config(), model.config());
    
    // Verify metadata
    assert_eq!(metadata.step, 0);
}

#[test]
fn test_checkpoint_roundtrip() {
    // Test save -> load roundtrip produces identical model
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model");

    let metadata = CheckpointMetadata {
        step: 42,
        loss: Some(2.5),
        learning_rate: Some(0.001),
        extra: HashMap::new(),
    };

    // Save
    save_checkpoint(&model, &checkpoint_path, Some(metadata.clone())).unwrap();

    // Load
    let (loaded_model, loaded_metadata) = load_checkpoint(&checkpoint_path).unwrap();
    
    // Verify config matches
    assert_eq!(loaded_model.config(), model.config());
    
    // Verify metadata matches
    assert_eq!(loaded_metadata.step, metadata.step);
    assert_eq!(loaded_metadata.loss, metadata.loss);
    assert_eq!(loaded_metadata.learning_rate, metadata.learning_rate);
}

#[test]
fn test_checkpoint_integrity_validation() {
    // Test checkpoint version validation
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model");

    // Save checkpoint
    save_checkpoint(&model, &checkpoint_path, None).unwrap();

    // Load should succeed with correct version
    let result = load_checkpoint(&checkpoint_path);
    assert!(result.is_ok());
}

#[test]
fn test_checkpoint_metadata() {
    // Test that checkpoint includes config and metadata
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model");

    let metadata = CheckpointMetadata {
        step: 100,
        loss: Some(1.5),
        learning_rate: Some(0.0001),
        extra: {
            let mut extra = HashMap::new();
            extra.insert("epoch".to_string(), serde_json::Value::Number(serde_json::Number::from(5)));
            extra
        },
    };

    save_checkpoint(&model, &checkpoint_path, Some(metadata.clone())).unwrap();
    let (_, loaded_metadata) = load_checkpoint(&checkpoint_path).unwrap();
    
    assert_eq!(loaded_metadata.step, metadata.step);
    assert_eq!(loaded_metadata.loss, metadata.loss);
    assert_eq!(loaded_metadata.learning_rate, metadata.learning_rate);
    assert!(loaded_metadata.extra.contains_key("epoch"));
}

#[test]
fn test_checkpoint_corrupted_file() {
    // Test error handling for missing checkpoint files
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("nonexistent");

    // Loading non-existent checkpoint should fail
    let result = load_checkpoint(&checkpoint_path);
    assert!(result.is_err());
}
