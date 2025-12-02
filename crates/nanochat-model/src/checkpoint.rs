//! Checkpoint save/load functionality
//!
//! This module provides functionality to save and load model checkpoints in `.apr` format.
//! Checkpoints include model weights, configuration, and metadata.

use crate::{GPT, GPTConfig};
use aprender::nn::serialize::{state_dict, StateDict, save_model, load_model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use anyhow::{Context, Result};

/// Checkpoint metadata containing training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Training step number
    pub step: usize,
    /// Loss value at this checkpoint
    pub loss: Option<f32>,
    /// Learning rate at this checkpoint
    pub learning_rate: Option<f32>,
    /// Additional metadata as key-value pairs
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            step: 0,
            loss: None,
            learning_rate: None,
            extra: HashMap::new(),
        }
    }
}

/// Checkpoint format version for compatibility checking
const CHECKPOINT_VERSION: &str = "1.0.0";

/// Save a GPT model checkpoint to disk
///
/// # Arguments
/// * `model` - The GPT model to save
/// * `path` - Path to save the checkpoint (directory will be created if needed)
/// * `metadata` - Optional checkpoint metadata
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
/// Returns an error if the directory cannot be created, weights cannot be serialized,
/// or the checkpoint file cannot be written.
pub fn save_checkpoint<P: AsRef<Path>>(
    model: &GPT,
    path: P,
    metadata: Option<CheckpointMetadata>,
) -> Result<()> {
    let path = path.as_ref();
    
    // Create directory if it doesn't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create checkpoint directory: {}", parent.display()))?;
    }

    // Save weights to SafeTensors format (aprender's standard)
    let weights_path = path.with_extension("safetensors");
    save_model(model, &weights_path)
        .map_err(|e| anyhow::anyhow!("Failed to save weights to SafeTensors: {}", e))?;

    // Save metadata (config, version, training info) to JSON
    let metadata_path = path.with_extension("json");
    let metadata_data = CheckpointMetadata {
        step: metadata.as_ref().map(|m| m.step).unwrap_or(0),
        loss: metadata.as_ref().and_then(|m| m.loss),
        learning_rate: metadata.as_ref().and_then(|m| m.learning_rate),
        extra: {
            let mut extra = HashMap::new();
            extra.insert("version".to_string(), serde_json::Value::String(CHECKPOINT_VERSION.to_string()));
            extra.insert("config".to_string(), serde_json::to_value(model.config())?);
            if let Some(m) = metadata {
                extra.extend(m.extra);
            }
            extra
        },
    };
    let json_data = serde_json::to_string_pretty(&metadata_data)
        .context("Failed to serialize metadata to JSON")?;
    fs::write(&metadata_path, json_data)
        .with_context(|| format!("Failed to write metadata file: {}", metadata_path.display()))?;

    Ok(())
}

/// Load a GPT model checkpoint from disk
///
/// # Arguments
/// * `path` - Path to the checkpoint file
///
/// # Returns
/// Tuple of (model, metadata)
///
/// # Errors
/// Returns an error if the checkpoint file cannot be read, parsed, or validated.
pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<(GPT, CheckpointMetadata)> {
    let path = path.as_ref();
    
    // Load metadata from JSON
    let metadata_path = path.with_extension("json");
    let json_data = fs::read_to_string(&metadata_path)
        .with_context(|| format!("Failed to read metadata file: {}", metadata_path.display()))?;
    
    let metadata: CheckpointMetadata = serde_json::from_str(&json_data)
        .context("Failed to parse metadata JSON")?;

    // Extract config from metadata
    let config_value = metadata.extra.get("config")
        .ok_or_else(|| anyhow::anyhow!("Missing config in metadata"))?;
    let config: GPTConfig = serde_json::from_value(config_value.clone())
        .context("Failed to parse config from metadata")?;

    // Validate version
    let version = metadata.extra.get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing version in metadata"))?;
    if version != CHECKPOINT_VERSION {
        anyhow::bail!(
            "Checkpoint version mismatch: expected {}, got {}",
            CHECKPOINT_VERSION,
            version
        );
    }

    // Create model from config
    let mut model = GPT::new(config);

    // Load weights from SafeTensors
    let weights_path = path.with_extension("safetensors");
    load_model(&mut model, &weights_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights from SafeTensors: {}", e))?;

    Ok((model, metadata))
}

// CheckpointData is no longer needed - we use SafeTensors for weights and JSON for metadata separately

/// Extract weights from a GPT model using aprender's state_dict
fn extract_weights(_model: &GPT) -> Result<StateDict> {
    // Use aprender's state_dict to extract all parameters
    // Note: This is currently unused as we use save_model directly
    // but kept for potential future use
    Ok(state_dict(_model, ""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_save_checkpoint() {
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
    fn test_load_checkpoint() {
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

        // Save with metadata
        save_checkpoint(&model, &checkpoint_path, Some(metadata.clone())).unwrap();

        // Load and verify
        let (_, loaded_metadata) = load_checkpoint(&checkpoint_path).unwrap();
        assert_eq!(loaded_metadata.step, metadata.step);
        assert_eq!(loaded_metadata.loss, metadata.loss);
        assert_eq!(loaded_metadata.learning_rate, metadata.learning_rate);
    }

    #[test]
    fn test_checkpoint_checksum_validation() {
        // Test version validation (checksum removed in favor of SafeTensors format)
        let config = GPTConfig::default();
        let model = GPT::new(config);
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("model");

        save_checkpoint(&model, &checkpoint_path, None).unwrap();

        // Load should succeed with valid checkpoint
        let result = load_checkpoint(&checkpoint_path);
        assert!(result.is_ok());

        // Corrupt SafeTensors file
        let safetensors_path = checkpoint_path.with_extension("safetensors");
        fs::write(&safetensors_path, b"corrupted").unwrap();

        // Load should fail with corrupted weights
        let result = load_checkpoint(&checkpoint_path);
        assert!(result.is_err());
    }
}
