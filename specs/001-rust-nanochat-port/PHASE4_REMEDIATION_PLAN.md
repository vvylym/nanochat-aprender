# Phase 4 Pretraining: Complete Remediation Plan

**Date**: 2024-12-19  
**Status**: Draft  
**Priority**: All issues (P1-P18) with concrete implementation steps

## Executive Summary

This document provides a complete remediation plan for all 18 identified issues in Phase 4 (Pretraining) implementation. Issues are organized by priority (CRITICAL → HIGH → MEDIUM → LOW) with concrete code changes, file paths, and implementation steps.

---

## CRITICAL Issues (Must Fix Before Production)

### P1: Target Shifting Not Implemented

**Severity**: CRITICAL  
**Location**: `crates/nanochat-pretrain/src/train.rs:99-100`  
**Impact**: Model learns wrong task (predicts current token instead of next token)

**Current Code**:
```rust
// Create targets (shifted input for next token prediction)
// For language modeling, targets are input shifted by 1 position
// For now, use the same as input (simplified - should be shifted in production)
let targets = batch.clone();
```

**Python Reference**: `dataloader.py:76-77`
```python
inputs_cpu = scratch[:-1]
targets_cpu = scratch[1:]
```

**Remediation**:

**Option A: Fix in DataLoader (Recommended)**
- Modify `DataLoader::next_batch()` to return both inputs and targets
- This matches Python's approach where dataloader handles the shifting

**File**: `crates/nanochat-pretrain/src/dataloader.rs`

```rust
/// Get the next batch (inputs and targets)
///
/// Returns a tuple of (inputs, targets) tensors, both of shape [batch_size, seq_len].
/// Targets are inputs shifted by 1 position for next-token prediction.
/// Returns None when all data has been consumed.
pub fn next_batch(&mut self) -> Result<Option<(Tensor, Tensor)>> {
    // Check if we have enough tokens for a batch (+1 for target)
    let tokens_needed = self.batch_size * (self.seq_len + 1);
    if self.current_pos + tokens_needed > self.token_ids.len() {
        // Reset position and shuffle for next epoch
        self.current_pos = 0;
        self.shuffle();

        // Check again after shuffle
        if self.current_pos + tokens_needed > self.token_ids.len() {
            return Ok(None);
        }
    }

    // Extract tokens for inputs and targets
    let mut inputs_data = Vec::new();
    let mut targets_data = Vec::new();
    
    for _ in 0..self.batch_size {
        let start = self.current_pos;
        let end = (start + self.seq_len + 1).min(self.token_ids.len());

        if end - start < self.seq_len + 1 {
            // Not enough tokens, pad with 0
            let mut seq = self.token_ids[start..end].to_vec();
            seq.resize(self.seq_len + 1, 0);
            
            // Split into inputs (first seq_len) and targets (last seq_len)
            inputs_data.extend(seq[..self.seq_len].iter().map(|&id| id as f32));
            targets_data.extend(seq[1..].iter().map(|&id| id as f32));
        } else {
            // Extract inputs: [start, start+seq_len)
            inputs_data.extend(self.token_ids[start..start+self.seq_len].iter().map(|&id| id as f32));
            // Extract targets: [start+1, start+seq_len+1) - shifted by 1
            targets_data.extend(self.token_ids[start+1..start+self.seq_len+1].iter().map(|&id| id as f32));
        }

        self.current_pos = end;
    }

    // Create tensors
    let inputs = Tensor::new(&inputs_data, &[self.batch_size, self.seq_len])?;
    let targets = Tensor::new(&targets_data, &[self.batch_size, self.seq_len])?;
    
    Ok(Some((inputs, targets)))
}
```

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
// Update training loop to use new DataLoader API
while step < training_config.max_steps {
    // Get next batch (now returns inputs and targets)
    let (batch, targets) = match dataloader.next_batch()? {
        Some((b, t)) => (b, t),
        None => {
            // Epoch finished, reset dataloader
            dataloader.reset();
            continue;
        }
    };

    // Forward pass and backward pass (gradient accumulation)
    let loss = model
        .forward_training(&batch, &targets, None)
        .context("Forward training failed")?;
    
    // ... rest of training loop
}
```

**Option B: Fix in Training Loop (Alternative)**
- Keep DataLoader as-is, shift targets in training loop

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
// Create targets (shifted input for next token prediction)
// For language modeling, targets are input shifted by 1 position
let batch_shape = batch.shape();
let batch_size = batch_shape[0];
let seq_len = batch_shape[1];

// Get batch data and shift by 1 position
let batch_data = batch.data();
let mut targets_data = Vec::with_capacity(batch_size * seq_len);

for i in 0..batch_size {
    let start = i * seq_len;
    // For each sequence, targets are inputs shifted by 1
    // First token of targets = second token of inputs
    if seq_len > 1 {
        targets_data.extend_from_slice(&batch_data[start + 1..start + seq_len]);
        // Last token of targets = padding (0) or next sequence's first token
        targets_data.push(0.0); // Simplified - should handle properly
    } else {
        targets_data.push(0.0);
    }
}

let targets = Tensor::new(&targets_data, &[batch_size, seq_len])?;
```

**Recommendation**: Use Option A (fix in DataLoader) as it matches Python's design and is cleaner.

**Testing**:
- Add test: `test_dataloader_target_shifting()` in `test_dataloader.rs`
- Verify targets[i] == inputs[i+1] for all positions

---

## HIGH Priority Issues (Required for Production)

### P2: Config File Loading

**Severity**: HIGH  
**Location**: `crates/nanochat-pretrain/src/main.rs:88-136`  
**Impact**: 12 hardcoded values prevent production use

**Remediation**:

**Step 1**: Create config structure

**File**: `crates/nanochat-pretrain/src/config.rs` (NEW)

```rust
//! Training configuration structures

use serde::{Deserialize, Serialize};
use std::path::Path;
use anyhow::{Context, Result};

/// Complete training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigFile {
    /// Model configuration
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingHyperparams,
    /// Optimizer configuration
    pub optimizer: OptimizerHyperparams,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size (must match tokenizer)
    pub vocab_size: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Number of key-value heads (GQA)
    pub n_kv_head: usize,
    /// Model embedding dimension
    pub n_embd: usize,
    /// Maximum sequence length
    pub sequence_len: usize,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparams {
    /// Batch size per device
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum training steps
    pub max_steps: usize,
    /// Checkpoint save interval (steps)
    pub save_interval: usize,
    /// Logging interval (steps)
    pub log_interval: usize,
    /// Validation evaluation interval (steps, 0 = disabled)
    pub eval_interval: usize,
    /// Number of tokens for validation evaluation
    pub eval_tokens: usize,
    /// Gradient clipping threshold (0.0 = disabled)
    pub grad_clip: f32,
}

/// Optimizer hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerHyperparams {
    /// Learning rate
    pub learning_rate: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// AdamW beta1
    pub beta1: f32,
    /// AdamW beta2
    pub beta2: f32,
    /// AdamW epsilon
    pub eps: f32,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Maximum steps (for scheduler)
    pub max_steps: usize,
    /// Minimum learning rate (for cosine decay)
    pub min_lr: f32,
    /// Warmup ratio (alternative to warmup_steps)
    pub warmup_ratio: Option<f32>,
    /// Warmdown ratio (for cosine decay)
    pub warmdown_ratio: Option<f32>,
    /// Final LR fraction (for cosine decay)
    pub final_lr_frac: Option<f32>,
}

impl TrainingConfigFile {
    /// Load configuration from JSON file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;
        let config: TrainingConfigFile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path))?;
        Ok(config)
    }

    /// Create default configuration
    pub fn default() -> Self {
        Self {
            model: ModelConfig {
                vocab_size: 50304,
                n_layer: 12,
                n_head: 6,
                n_kv_head: 6,
                n_embd: 768,
                sequence_len: 1024,
            },
            training: TrainingHyperparams {
                batch_size: 32,
                seq_len: 256,
                gradient_accumulation_steps: 1,
                max_steps: 10000,
                save_interval: 1000,
                log_interval: 100,
                eval_interval: 250,
                eval_tokens: 20 * 524288,
                grad_clip: 1.0,
            },
            optimizer: OptimizerHyperparams {
                learning_rate: 1e-4,
                weight_decay: 0.1,
                beta1: 0.9,
                beta2: 0.95,
                eps: 1e-8,
                warmup_steps: 1000,
                max_steps: 10000,
                min_lr: 1e-6,
                warmup_ratio: None,
                warmdown_ratio: Some(0.2),
                final_lr_frac: Some(0.0),
            },
        }
    }
}
```

**Step 2**: Update `Cargo.toml` to add `serde` dependencies

**File**: `crates/nanochat-pretrain/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Step 3**: Update `main.rs` to use config

**File**: `crates/nanochat-pretrain/src/main.rs`

```rust
use nanochat_pretrain::config::TrainingConfigFile;

fn main() -> Result<()> {
    let args = Args::parse();

    // Load configuration
    let config = if let Some(config_path) = &args.config {
        TrainingConfigFile::from_file(config_path)
            .context("Failed to load config file")?
    } else {
        // Use defaults if no config provided
        TrainingConfigFile::default()
    };

    // Create model config from loaded config
    let model_config = GPTConfig {
        vocab_size: config.model.vocab_size,
        n_layer: config.model.n_layer,
        n_head: config.model.n_head,
        n_kv_head: config.model.n_kv_head,
        n_embd: config.model.n_embd,
        sequence_len: config.model.sequence_len,
        dropout: 0.0, // TODO: Add to config if needed
    };

    // Create training config
    let training_config = TrainingConfig {
        batch_size: config.training.batch_size,
        seq_len: config.training.seq_len,
        gradient_accumulation_steps: config.training.gradient_accumulation_steps,
        max_steps: config.training.max_steps,
        save_interval: config.training.save_interval,
        log_interval: config.training.log_interval,
    };

    // Create optimizer config
    let optimizer_config = OptimizerConfig {
        learning_rate: config.optimizer.learning_rate,
        weight_decay: config.optimizer.weight_decay,
        beta1: config.optimizer.beta1,
        beta2: config.optimizer.beta2,
        eps: config.optimizer.eps,
        warmup_steps: config.optimizer.warmup_steps,
        max_steps: config.optimizer.max_steps,
        min_lr: config.optimizer.min_lr,
    };

    // ... rest of main
}
```

**Step 4**: Create example config file

**File**: `crates/nanochat-pretrain/config.example.json`

```json
{
  "model": {
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 6,
    "n_kv_head": 6,
    "n_embd": 768,
    "sequence_len": 1024
  },
  "training": {
    "batch_size": 32,
    "seq_len": 256,
    "gradient_accumulation_steps": 1,
    "max_steps": 10000,
    "save_interval": 1000,
    "log_interval": 100,
    "eval_interval": 250,
    "eval_tokens": 10485760,
    "grad_clip": 1.0
  },
  "optimizer": {
    "learning_rate": 0.0001,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "warmup_steps": 1000,
    "max_steps": 10000,
    "min_lr": 1e-6,
    "warmup_ratio": null,
    "warmdown_ratio": 0.2,
    "final_lr_frac": 0.0
  }
}
```

**Step 5**: Update `lib.rs` to export config module

**File**: `crates/nanochat-pretrain/src/lib.rs`

```rust
pub mod config;
pub mod dataloader;
pub mod metrics;
pub mod optimizer;
pub mod train;
```

**Testing**:
- Add test: `test_config_loading()` in `tests/test_config.rs`
- Verify all fields are loaded correctly
- Test with missing fields (should use defaults or error)

---

### P3: Tokenizer Training from Data Files

**Severity**: HIGH  
**Location**: `crates/nanochat-pretrain/src/main.rs:166`  
**Impact**: Cannot train tokenizer from real data

**Remediation**:

**File**: `crates/nanochat-tokenizer/src/lib.rs`

Add method to `Tokenizer`:

```rust
impl Tokenizer {
    /// Train tokenizer from text files in a directory
    ///
    /// # Arguments
    /// * `data_dir` - Directory containing text files (.txt)
    /// * `vocab_size` - Target vocabulary size
    /// * `special_tokens` - Optional special tokens to add
    ///
    /// # Returns
    /// Trained tokenizer
    pub fn train_from_directory(
        data_dir: &Path,
        vocab_size: usize,
        special_tokens: Option<Vec<String>>,
    ) -> Result<Self> {
        use std::fs;
        use std::io::Read;

        // Collect all text from .txt files
        let mut texts = Vec::new();
        let entries = fs::read_dir(data_dir)
            .with_context(|| format!("Failed to read directory: {:?}", data_dir))?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                let mut file = fs::File::open(&path)
                    .with_context(|| format!("Failed to open file: {:?}", path))?;
                let mut content = String::new();
                file.read_to_string(&mut content)
                    .with_context(|| format!("Failed to read file: {:?}", path))?;
                texts.push(content);
            }
        }

        if texts.is_empty() {
            anyhow::bail!("No .txt files found in directory: {:?}", data_dir);
        }

        // Create iterator over texts
        let text_iter = texts.iter().map(|s| s.as_str());

        // Train tokenizer
        let mut bpe = aprender::text::tokenize::BpeTokenizer::train(
            text_iter,
            vocab_size,
        )?;

        // Add special tokens if provided
        if let Some(special) = special_tokens {
            for token in special {
                bpe.add_special_token(&token)?;
            }
        }

        Ok(Self { bpe })
    }
}
```

**File**: `crates/nanochat-pretrain/src/main.rs`

```rust
fn create_or_load_tokenizer(data_dir: &PathBuf) -> Result<Tokenizer> {
    // Check if tokenizer already exists
    let tokenizer_path = data_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        // Load existing tokenizer
        Tokenizer::from_directory(data_dir)
            .with_context(|| format!("Failed to load tokenizer from: {:?}", data_dir))
    } else {
        // Train new tokenizer from data files
        let vocab_size = 50000; // TODO: Load from config
        let special_tokens = vec![
            "<|bos|>".to_string(),
            "<|eos|>".to_string(),
            "<|pad|>".to_string(),
        ];
        
        Tokenizer::train_from_directory(data_dir, vocab_size, Some(special_tokens))
            .context("Failed to train tokenizer from data files")
    }
}
```

**Testing**:
- Add test: `test_tokenizer_training_from_directory()` in `nanochat-tokenizer/tests/`
- Verify tokenizer can be trained from multiple files
- Verify special tokens are added correctly

---

### P4: Gradient Clipping

**Severity**: HIGH  
**Location**: `crates/nanochat-pretrain/src/train.rs` (missing)  
**Impact**: Training instability on large gradients

**Remediation**:

**Step 1**: Check if aprender provides gradient clipping

If aprender doesn't provide it, implement manually:

**File**: `crates/nanochat-pretrain/src/train.rs`

Add function:

```rust
/// Clip gradients by norm
///
/// # Arguments
/// * `model` - The GPT model
/// * `max_norm` - Maximum gradient norm (0.0 = disabled)
///
/// # Returns
/// Gradient norm (before clipping)
fn clip_gradients(model: &GPT, max_norm: f32) -> Result<f32> {
    if max_norm <= 0.0 {
        return Ok(0.0); // Clipping disabled
    }

    use aprender::nn::Module;
    
    // Compute gradient norm
    let parameters = model.parameters();
    let mut total_norm_sq = 0.0;
    
    for param in parameters {
        if let Some(grad) = param.grad() {
            let grad_data = grad.data();
            for &g in grad_data {
                total_norm_sq += g * g;
            }
        }
    }
    
    let total_norm = total_norm_sq.sqrt();
    
    // Clip if norm exceeds threshold
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad_mut() {
                let grad_data = grad.data_mut();
                for g in grad_data {
                    *g *= clip_coef;
                }
            }
        }
    }
    
    Ok(total_norm)
}
```

**Step 2**: Add `grad_clip` to `TrainingConfig`

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    // ... existing fields ...
    /// Gradient clipping threshold (0.0 = disabled)
    pub grad_clip: f32,
}
```

**Step 3**: Use in training loop

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
// Only update optimizer after accumulation steps
if accumulation_count >= training_config.gradient_accumulation_steps {
    // Gradient clipping (before optimizer step)
    let grad_norm = if training_config.grad_clip > 0.0 {
        clip_gradients(&model, training_config.grad_clip)
            .context("Failed to clip gradients")?
    } else {
        0.0
    };

    // Optimizer step: update parameters using accumulated gradients
    optimizer.step();
    
    // ... rest of loop
}
```

**Step 4**: Add to metrics logging

**File**: `crates/nanochat-pretrain/src/metrics.rs`

```rust
pub struct TrainingMetrics {
    // ... existing fields ...
    /// Gradient norm (if clipping enabled)
    pub grad_norm: Option<f32>,
}
```

**Testing**:
- Add test: `test_gradient_clipping()` in `tests/test_train.rs`
- Verify gradients are clipped when norm exceeds threshold
- Verify no clipping when `grad_clip = 0.0`

---

### P5: Validation Loss Evaluation

**Severity**: HIGH  
**Location**: `crates/nanochat-pretrain/src/train.rs` (missing)  
**Impact**: No validation monitoring during training

**Remediation**:

**Step 1**: Create validation dataloader

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
/// Evaluate validation loss
///
/// # Arguments
/// * `model` - The GPT model (in eval mode)
/// * `val_dataloader` - Validation data loader
/// * `num_steps` - Number of batches to evaluate
///
/// # Returns
/// Average validation loss (bits per byte)
fn evaluate_validation_loss(
    model: &GPT,
    val_dataloader: &mut DataLoader,
    num_steps: usize,
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut total_tokens = 0;

    for _ in 0..num_steps {
        match val_dataloader.next_batch()? {
            Some((batch, targets)) => {
                // Forward pass (no gradients)
                let loss = model.forward_training(&batch, &targets, None)
                    .context("Validation forward pass failed")?;
                
                let batch_size = batch.shape()[0];
                let seq_len = batch.shape()[1];
                total_tokens += batch_size * seq_len;
                total_loss += loss.item() * (batch_size * seq_len) as f32;
            }
            None => break, // No more validation data
        }
    }

    if total_tokens == 0 {
        anyhow::bail!("No validation data available");
    }

    // Convert loss to bits per byte (bpb)
    // CrossEntropyLoss gives nats, convert to bits: nats * log2(e) = nats * 1.442695
    let avg_loss = total_loss / total_tokens as f32;
    let bpb = avg_loss * 1.442695;

    Ok(bpb)
}
```

**Step 2**: Update `train()` function signature

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
pub fn train(
    mut model: GPT,
    mut dataloader: DataLoader,
    training_config: TrainingConfig,
    optimizer_config: OptimizerConfig,
    output_dir: &Path,
    val_dataloader: Option<&mut DataLoader>, // Add validation dataloader
) -> Result<()> {
    // ... existing setup ...

    let mut min_val_bpb = f32::INFINITY;

    while step < training_config.max_steps {
        // ... training loop ...

        // Validation evaluation
        if let Some(ref mut val_loader) = val_dataloader {
            if training_config.eval_interval > 0 && step % training_config.eval_interval == 0 {
                let val_bpb = evaluate_validation_loss(&model, val_loader, training_config.eval_steps)
                    .context("Validation evaluation failed")?;
                
                if val_bpb < min_val_bpb {
                    min_val_bpb = val_bpb;
                }

                println!("Step {:05} | Validation bpb: {:.4}", step, val_bpb);
                
                // Log to metrics
                // TODO: Add validation metrics to MetricsLogger
            }
        }

        step += 1;
    }

    Ok(())
}
```

**Step 3**: Add `eval_interval` and `eval_steps` to `TrainingConfig`

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    // ... existing fields ...
    /// Validation evaluation interval (0 = disabled)
    pub eval_interval: usize,
    /// Number of validation batches to evaluate
    pub eval_steps: usize,
}
```

**Step 4**: Update `main.rs` to create validation dataloader

**File**: `crates/nanochat-pretrain/src/main.rs`

```rust
// Create validation data loader (if validation data exists)
let val_data_dir = args.data_dir.join("val");
let val_dataloader = if val_data_dir.exists() {
    let val_tokenizer = tokenizer.clone(); // Clone tokenizer for validation
    Some(DataLoader::new(
        &val_data_dir,
        val_tokenizer,
        training_config.batch_size,
        training_config.seq_len,
        args.workers,
    )?)
} else {
    None
};
```

**Testing**:
- Add test: `test_validation_evaluation()` in `tests/test_train.rs`
- Verify validation loss is computed correctly
- Verify `min_val_bpb` tracking works

---

## MEDIUM Priority Issues

### P6: Optimizer State Checkpointing

**Severity**: MEDIUM  
**Location**: `crates/nanochat-pretrain/src/train.rs` (checkpoint functions)  
**Impact**: Cannot properly resume training (optimizer state lost)

**Remediation**:

**Step 1**: Extend `CheckpointMetadata` to include optimizer state

**File**: `crates/nanochat-model/src/checkpoint.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    // ... existing fields ...
    /// Optimizer state (serialized)
    pub optimizer_state: Option<serde_json::Value>,
}
```

**Step 2**: Save optimizer state in checkpoint

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
// Need to serialize optimizer state
// aprender's AdamW may need custom serialization
// For now, save optimizer config and step count

fn save_checkpoint_step(
    model: &GPT,
    optimizer: &AdamW, // Add optimizer parameter
    output_dir: &Path,
    step: usize,
    loss: Option<f32>,
    lr: Option<f32>,
) -> Result<()> {
    // ... existing code ...

    // Serialize optimizer state (if aprender supports it)
    // Otherwise, save optimizer config and step
    let optimizer_state = serde_json::json!({
        "step": step,
        "lr": optimizer.lr(),
        // Add more optimizer state if aprender provides serialization
    });

    let metadata = CheckpointMetadata {
        step,
        loss,
        learning_rate: lr,
        optimizer_state: Some(optimizer_state),
        extra: std::collections::HashMap::new(),
    };

    // ... save checkpoint ...
}
```

**Step 3**: Load optimizer state on resume

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
pub fn resume_from_checkpoint(
    checkpoint_path: &Path,
    optimizer_config: OptimizerConfig,
) -> Result<(GPT, AdamW, CheckpointMetadata, usize)> {
    let (model, metadata) = load_checkpoint(checkpoint_path)?;
    
    // Recreate optimizer
    let mut temp_model = model.clone(); // Temporary for optimizer setup
    let (mut optimizer, mut scheduler) = setup_optimizers(&mut temp_model, optimizer_config)?;
    
    // Restore optimizer state if available
    if let Some(opt_state) = &metadata.optimizer_state {
        // Deserialize and restore optimizer state
        // This depends on aprender's optimizer serialization API
    }
    
    Ok((model, optimizer, metadata, metadata.step))
}
```

**Note**: This depends on aprender's optimizer serialization capabilities. May need to check aprender API.

---

### P7: DataLoader State Checkpointing

**Severity**: MEDIUM  
**Location**: `crates/nanochat-pretrain/src/dataloader.rs`  
**Impact**: Approximate resume (may skip some data)

**Remediation**:

**File**: `crates/nanochat-pretrain/src/dataloader.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderState {
    /// Current position in token stream
    pub current_pos: usize,
    /// RNG seed (for reproducibility)
    pub rng_seed: u64,
}

impl DataLoader {
    /// Get current state for checkpointing
    pub fn get_state(&self) -> DataLoaderState {
        DataLoaderState {
            current_pos: self.current_pos,
            rng_seed: 42, // TODO: Store actual seed
        }
    }

    /// Restore state from checkpoint
    pub fn restore_state(&mut self, state: DataLoaderState) {
        self.current_pos = state.current_pos;
        self.rng = StdRng::seed_from_u64(state.rng_seed);
    }
}
```

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
// Save dataloader state in checkpoint metadata
let dataloader_state = dataloader.get_state();
metadata.extra.insert(
    "dataloader_state".to_string(),
    serde_json::to_value(dataloader_state)?,
);

// On resume, restore dataloader state
if let Some(state_val) = metadata.extra.get("dataloader_state") {
    let state: DataLoaderState = serde_json::from_value(state_val.clone())?;
    dataloader.restore_state(state);
}
```

---

### P8: Enhanced LR Scheduling

**Severity**: MEDIUM  
**Location**: `crates/nanochat-pretrain/src/optimizer.rs`  
**Impact**: May not match Python's warmup/warmdown behavior

**Remediation**:

**File**: `crates/nanochat-pretrain/src/optimizer.rs`

```rust
/// Learning rate multiplier function (matching Python's get_lr_multiplier)
///
/// Python reference: base_train.py:180-189
fn get_lr_multiplier(step: usize, warmup_steps: usize, warmdown_steps: usize, total_steps: usize, final_lr_frac: f32) -> f32 {
    if step < warmup_steps {
        // Warmup: linear increase from 0 to 1
        (step + 1) as f32 / warmup_steps as f32
    } else if step <= total_steps - warmdown_steps {
        // Constant: 1.0
        1.0
    } else {
        // Warmdown: linear decrease from 1.0 to final_lr_frac
        let progress = (total_steps - step) as f32 / warmdown_steps as f32;
        progress * 1.0 + (1.0 - progress) * final_lr_frac
    }
}

/// Update learning rate using custom scheduler (if needed)
pub fn update_learning_rate_custom(
    scheduler: &mut WarmupCosineScheduler,
    optimizer: &mut AdamW,
    step: usize,
    config: &OptimizerConfig,
) {
    // Use aprender's scheduler if it matches Python behavior
    // Otherwise, compute multiplier manually
    if let (Some(warmup_ratio), Some(warmdown_ratio), Some(final_lr_frac)) = 
        (config.warmup_ratio, config.warmdown_ratio, config.final_lr_frac) {
        let warmup_steps = (warmup_ratio * config.max_steps as f32) as usize;
        let warmdown_steps = (warmdown_ratio * config.max_steps as f32) as usize;
        let lr_mult = get_lr_multiplier(step, warmup_steps, warmdown_steps, config.max_steps, final_lr_frac);
        
        // Manually set learning rate
        // This depends on aprender's optimizer API
        // optimizer.set_lr(config.learning_rate * lr_mult);
    } else {
        // Use aprender's scheduler
        scheduler.step(optimizer);
    }
}
```

---

### P9: Dual Optimizer Support (Optional)

**Severity**: MEDIUM  
**Location**: `crates/nanochat-pretrain/src/optimizer.rs`  
**Impact**: Python uses AdamW for embeddings + Muon for matrices

**Note**: This is complex and may not be necessary if single optimizer works. Defer unless needed.

**Remediation**: (Deferred - requires Muon optimizer implementation or aprender equivalent)

---

## LOW Priority Issues

### P10: EMA Loss Smoothing

**File**: `crates/nanochat-pretrain/src/metrics.rs`

```rust
pub struct MetricsLogger {
    // ... existing fields ...
    smooth_loss: f32,
    ema_beta: f32,
}

impl MetricsLogger {
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            step: 0,
            smooth_loss: 0.0,
            ema_beta: 0.9, // Python: ema_beta = 0.9
        }
    }

    pub fn log_step(&mut self, loss: &Tensor, learning_rate: f32, tokens_processed: usize, time_elapsed: f32) {
        self.step += 1;
        let loss_value = loss.item();
        
        // EMA smoothing
        self.smooth_loss = self.ema_beta * self.smooth_loss + (1.0 - self.ema_beta) * loss_value;
        
        // Debiased EMA (Python: debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)))
        let debiased_loss = self.smooth_loss / (1.0 - self.ema_beta.powi((self.step + 1) as i32));

        if self.step.is_multiple_of(self.log_interval) {
            println!(
                "Step {}: loss={:.6} (smooth={:.6}), lr={:.2e}, throughput={:.2} tokens/s",
                self.step, loss_value, debiased_loss, learning_rate, 
                if time_elapsed > 0.0 { tokens_processed as f32 / time_elapsed } else { 0.0 }
            );
        }
    }
}
```

---

### P11: Training Time Tracking

**File**: `crates/nanochat-pretrain/src/train.rs`

```rust
use std::time::{Duration, Instant};

pub fn train(...) -> Result<()> {
    // ... existing setup ...
    
    let mut total_training_time = Duration::ZERO;
    let mut step_start_time = Instant::now();

    while step < training_config.max_steps {
        // ... training step ...

        if accumulation_count >= training_config.gradient_accumulation_steps {
            let step_duration = step_start_time.elapsed();
            
            // Only count time after first 10 steps (warmup)
            if step > 10 {
                total_training_time += step_duration;
            }
            
            let time_elapsed = step_duration.as_secs_f32();
            let tokens_per_sec = tokens_processed as f32 / time_elapsed;
            
            metrics_logger.log_step(&loss, learning_rate, tokens_processed, time_elapsed);
            
            step_start_time = Instant::now();
        }
    }

    println!("Total training time: {:.2}m", total_training_time.as_secs_f64() / 60.0);
    Ok(())
}
```

---

### P12-P15: MFU, Sampling, CORE Metric, WandB

**Status**: Deferred to future phases  
**Rationale**: Nice-to-have features that don't block core functionality

**Remediation**: (Documented for future implementation)

---

## Implementation Order

1. **P1** (CRITICAL): Target shifting - Blocks correct training
2. **P2** (HIGH): Config loading - Required for production
3. **P3** (HIGH): Tokenizer training - Required for real data
4. **P4** (HIGH): Gradient clipping - Important for stability
5. **P5** (HIGH): Validation loop - Required for monitoring
6. **P6** (MEDIUM): Optimizer state - Improves resume capability
7. **P7** (MEDIUM): DataLoader state - Improves resume accuracy
8. **P8** (MEDIUM): Enhanced LR scheduling - Matches Python behavior
9. **P10-P11** (LOW): Metrics enhancements - Quality of life
10. **P9, P12-P15** (LOW/OPTIONAL): Advanced features - Future work

---

## Testing Requirements

For each remediation:

1. **Unit Tests**: Test the specific functionality in isolation
2. **Integration Tests**: Test within the training loop
3. **Regression Tests**: Verify existing functionality still works
4. **Quality Gates**: `cargo fmt`, `cargo clippy`, `cargo test --all-features`

---

## New Artifact Proposal: Task Completion Checklist

To prevent tasks being marked complete when gaps remain, propose a new artifact:

**File**: `.specify/templates/task-completion-checklist.md`

This checklist would be required before marking any task as complete, ensuring:
- No TODOs remain
- All placeholders are implemented
- Tests cover the functionality
- Documentation is complete
- Quality gates pass

See separate section below for full proposal.

