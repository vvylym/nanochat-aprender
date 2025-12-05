//! Training loop for pretraining

use crate::dataloader::{DataLoader, DataLoaderState};
use crate::metrics::MetricsLogger;
use crate::optimizer::{
    get_learning_rate, setup_optimizers, update_learning_rate_custom, OptimizerConfig,
};
use anyhow::{Context, Result};
use aprender::autograd::Tensor;
use aprender::nn::optim::{AdamW, Optimizer};
use nanochat_model::{
    checkpoint::{load_checkpoint, save_checkpoint, CheckpointMetadata},
    GPT,
};
use std::path::Path;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum number of training steps
    pub max_steps: usize,
    /// Checkpoint save interval (in steps)
    pub save_interval: usize,
    /// Logging interval (in steps)
    pub log_interval: usize,
    /// Gradient clipping threshold (0.0 = disabled)
    pub grad_clip: f32,
    /// Validation evaluation interval (0 = disabled)
    pub eval_interval: usize,
    /// Number of validation batches to evaluate
    pub eval_steps: usize,
}

/// Clip gradients by norm
///
/// Computes the gradient norm across all model parameters and clips gradients
/// if the norm exceeds the threshold. This prevents gradient explosion during training.
///
/// # Arguments
/// * `model` - The GPT model (mutable reference needed to modify gradients)
/// * `max_norm` - Maximum gradient norm (0.0 = disabled)
///
/// # Returns
/// Gradient norm (before clipping)
///
/// # Python Reference
/// Matches `base_train.py:316`:
/// ```python
/// grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
/// ```
///
/// # Note
/// This is a placeholder implementation. Gradient clipping requires access to gradient
/// values from aprender's computation graph. The exact API for accessing gradients
/// in aprender needs to be determined. For now, this function returns 0.0 (no clipping)
/// and logs a warning. This should be updated once aprender's gradient access API is
/// understood.
/// Clip gradients by global norm (matching PyTorch's `torch.nn.utils.clip_grad_norm_`)
///
/// **Current Implementation**: Computes and returns the global gradient norm for monitoring.
/// Actual gradient clipping (scaling) requires aprender API extension for setting gradients.
///
/// # Arguments
/// * `model` - The GPT model
/// * `max_norm` - Maximum gradient norm (clipping threshold)
///
/// # Returns
/// The global gradient norm (for logging/monitoring)
///
/// # Algorithm
/// 1. Compute global norm: sqrt(sum(grad²) for all parameters)
/// 2. If norm > max_norm: Log warning (clipping not yet implemented)
///
/// # Python Reference
/// Matches PyTorch's `torch.nn.utils.clip_grad_norm_(parameters, max_norm)`
///
/// # Limitations
/// - Gradient norm computation: ✅ Implemented
/// - Gradient clipping (scaling): ⏳ Requires aprender API for setting gradients
///   - See: `specs/001-rust-nanochat-port/APRENDER_API_SOLUTIONS.md`
#[allow(dead_code)] // Used in tests
pub fn clip_gradients(model: &GPT, max_norm: f32) -> Result<f32> {
    use aprender::autograd::get_grad;
    use aprender::nn::Module;

    if max_norm <= 0.0 {
        return Ok(0.0);
    }

    // Get all model parameters
    let parameters = model.parameters();

    // Compute global norm
    let mut total_norm_sq = 0.0;
    let mut param_count = 0;

    for param in parameters {
        if let Some(grad) = get_grad(param.id()) {
            let grad_data = grad.data();
            // Sum of squares for this parameter's gradient
            let param_norm_sq: f32 = grad_data.iter().map(|&x| x * x).sum();
            total_norm_sq += param_norm_sq;
            param_count += 1;
        }
    }

    let total_norm = total_norm_sq.sqrt();

    // Log warning if clipping would be needed
    if total_norm > max_norm {
        eprintln!(
            "Warning: Gradient norm ({:.4}) exceeds max_norm ({:.4}) for {} parameters. \
             Clipping not yet implemented - requires aprender API for setting gradients. \
             See: specs/001-rust-nanochat-port/APRENDER_API_SOLUTIONS.md",
            total_norm, max_norm, param_count
        );
    }

    Ok(total_norm)
}

/// Train a single step (forward + backward, no optimizer step)
///
/// This is used for gradient accumulation. The optimizer step is called
/// separately after accumulation is complete.
///
/// # Arguments
/// * `model` - The GPT model
/// * `idx` - Input token IDs [batch_size, seq_len]
/// * `targets` - Target token IDs [batch_size, seq_len]
///
/// # Returns
/// Loss tensor (scalar)
pub fn train_step(model: &mut GPT, idx: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Forward pass with targets to compute loss
    // This uses model's forward_training() which internally uses
    // aprender::nn::loss::CrossEntropyLoss per FR-087
    let loss = model.forward_training(idx, targets, None).context("Forward training failed")?;

    // Backward pass: compute gradients
    // aprender's loss tensor has backward() method for scalar tensors
    loss.backward();

    Ok(loss)
}

/// Evaluate validation loss
///
/// Computes average validation loss over a specified number of batches.
/// The model is evaluated in inference mode (no gradients computed).
///
/// # Arguments
/// * `model` - The GPT model (in eval mode, no gradients)
/// * `val_dataloader` - Validation data loader
/// * `num_steps` - Number of batches to evaluate
///
/// # Returns
/// Average validation loss (bits per byte)
///
/// # Python Reference
/// Matches `base_train.py:285-295`:
/// ```python
/// val_loss = evaluate_bpb(model, val_loader, eval_tokens)
/// ```
pub fn evaluate_validation_loss(
    model: &GPT,
    val_dataloader: &mut DataLoader,
    num_steps: usize,
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut total_tokens = 0;

    for _ in 0..num_steps {
        match val_dataloader.next_batch()? {
            Some((batch, targets)) => {
                // Forward pass (no gradients - model is not mutable)
                let loss = model
                    .forward_training(&batch, &targets, None)
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
    // CrossEntropyLoss gives nats, convert to bits: nats * log2(e)
    let avg_loss = total_loss / total_tokens as f32;
    let bpb = avg_loss * std::f32::consts::LOG2_E;

    Ok(bpb)
}

/// Run the full training loop
///
/// # Arguments
/// * `model` - The GPT model
/// * `dataloader` - Training data loader
/// * `training_config` - Training configuration
/// * `optimizer_config` - Optimizer configuration
/// * `output_dir` - Directory to save checkpoints
/// * `val_dataloader` - Optional validation data loader for monitoring
pub fn train(
    mut model: GPT,
    mut dataloader: DataLoader,
    training_config: TrainingConfig,
    optimizer_config: OptimizerConfig,
    output_dir: &Path,
    mut val_dataloader: Option<&mut DataLoader>,
) -> Result<()> {
    // Setup optimizers (requires mutable model reference)
    // Clone config since we need it later for custom LR scheduling
    let (mut optimizer, mut scheduler) = setup_optimizers(&mut model, optimizer_config.clone())
        .context("Failed to setup optimizers")?;

    // Create metrics logger
    let mut metrics_logger = MetricsLogger::new(training_config.log_interval);

    // Track minimum validation loss
    let mut min_val_bpb = f32::INFINITY;

    // Training loop
    let mut step = 0;
    let mut accumulated_loss = 0.0;
    let mut accumulation_count = 0;

    while step < training_config.max_steps {
        // Get next batch (now returns inputs and targets with proper shifting)
        let (batch, targets) = match dataloader.next_batch()? {
            Some((b, t)) => (b, t),
            None => {
                // Epoch finished, reset dataloader
                dataloader.reset();
                continue;
            }
        };

        // Forward pass and backward pass (gradient accumulation)
        // Targets are already properly shifted by 1 position in DataLoader
        let loss = model
            .forward_training(&batch, &targets, None)
            .context("Forward training failed")?;

        // Backward pass: compute gradients
        loss.backward();

        accumulated_loss += loss.item();
        accumulation_count += 1;

        // Only update optimizer after accumulation steps
        if accumulation_count >= training_config.gradient_accumulation_steps {
            // Gradient clipping (before optimizer step)
            // Note: Currently computes norm for monitoring; actual clipping requires aprender API
            if training_config.grad_clip > 0.0 {
                let _grad_norm = clip_gradients(&model, training_config.grad_clip)
                    .context("Failed to compute gradient norm")?;
                // Gradient norm is computed for monitoring; actual clipping requires aprender API
                // See: specs/001-rust-nanochat-port/APRENDER_API_SOLUTIONS.md
                // TODO: Use grad_norm for logging when metrics system supports it
            }

            // Optimizer step: update parameters using accumulated gradients
            optimizer.step();

            // Zero gradients for next iteration
            optimizer.zero_grad();

            // Update learning rate using scheduler
            // Use custom scheduler if warmdown_ratio is specified (matches Python behavior)
            // Otherwise use aprender's scheduler (cosine decay)
            update_learning_rate_custom(&mut scheduler, &mut optimizer, step, &optimizer_config);

            // Get current learning rate from optimizer
            let learning_rate = get_learning_rate(&optimizer);

            // Log metrics
            let avg_loss = accumulated_loss / accumulation_count as f32;
            let tokens_processed =
                training_config.batch_size * training_config.seq_len * accumulation_count;
            metrics_logger.log_step(&loss, learning_rate, tokens_processed, 1.0);

            // Save checkpoint at intervals
            if step > 0 && step % training_config.save_interval == 0 {
                save_checkpoint_step(
                    &model,
                    &optimizer,
                    &dataloader,
                    output_dir,
                    step,
                    Some(avg_loss),
                    Some(learning_rate),
                )
                .context("Failed to save checkpoint")?;
            }

            // Reset accumulation
            accumulated_loss = 0.0;
            accumulation_count = 0;
        }

        // Validation evaluation
        if let Some(ref mut val_loader) = val_dataloader {
            if training_config.eval_interval > 0
                && step > 0
                && step % training_config.eval_interval == 0
            {
                let val_bpb =
                    evaluate_validation_loss(&model, val_loader, training_config.eval_steps)
                        .context("Validation evaluation failed")?;

                if val_bpb < min_val_bpb {
                    min_val_bpb = val_bpb;
                }

                println!(
                    "Step {:05} | Validation bpb: {:.4} (min: {:.4})",
                    step, val_bpb, min_val_bpb
                );
            }
        }

        step += 1;
    }

    // Save final checkpoint
    if accumulation_count > 0 {
        let avg_loss = accumulated_loss / accumulation_count as f32;
        // Update learning rate one more time
        update_learning_rate_custom(&mut scheduler, &mut optimizer, step, &optimizer_config);
        let learning_rate = get_learning_rate(&optimizer);
        save_checkpoint_step(
            &model,
            &optimizer,
            &dataloader,
            output_dir,
            step,
            Some(avg_loss),
            Some(learning_rate),
        )
        .context("Failed to save final checkpoint")?;
    } else {
        let learning_rate = get_learning_rate(&optimizer);
        save_checkpoint_step(
            &model,
            &optimizer,
            &dataloader,
            output_dir,
            step,
            None,
            Some(learning_rate),
        )
        .context("Failed to save final checkpoint")?;
    }

    Ok(())
}

/// Save a training checkpoint
///
/// Saves model weights, optimizer state (step count and LR), and dataloader state
/// to allow resuming training from the exact same point.
///
/// # Arguments
/// * `model` - The GPT model
/// * `optimizer` - The optimizer (for saving step count and LR)
/// * `dataloader` - The data loader (for saving position and RNG seed)
/// * `output_dir` - Directory to save checkpoint
/// * `step` - Training step number
/// * `loss` - Optional loss value
/// * `lr` - Optional learning rate
///
/// # Note
/// Full optimizer state (moment estimates m, v) cannot be saved without aprender API support.
/// Only step count and learning rate are saved. On resume, optimizer will restart with
/// fresh moment estimates, which may cause a brief adjustment period.
///
/// See: `specs/001-rust-nanochat-port/APRENDER_API_SOLUTIONS.md` for details and
/// proposed aprender API extensions.
fn save_checkpoint_step(
    model: &GPT,
    optimizer: &AdamW,
    dataloader: &DataLoader,
    output_dir: &Path,
    step: usize,
    loss: Option<f32>,
    lr: Option<f32>,
) -> Result<()> {
    // Create checkpoint path (without extension, save_checkpoint will add .safetensors and .json)
    let checkpoint_path = output_dir.join(format!("checkpoint_step_{}", step));

    // Save optimizer state (what we can save without aprender API)
    // Note: aprender's AdamW doesn't expose moment estimates (m, v) for serialization
    // We save step count and LR, but full state restoration isn't possible
    // See: specs/001-rust-nanochat-port/APRENDER_API_SOLUTIONS.md for details
    let optimizer_state = serde_json::json!({
        "step": step,
        "lr": lr.unwrap_or_else(|| optimizer.lr()),
        // TODO: Add moment estimates (m, v) when aprender exposes them via get_state() API
        // See: specs/001-rust-nanochat-port/APRENDER_API_SOLUTIONS.md#limitation-2-optimizer-state-serialization-p6
    });

    // Save dataloader state
    let dataloader_state = dataloader.get_state();

    // Create metadata with optimizer and dataloader state
    let mut extra = std::collections::HashMap::new();
    extra.insert("optimizer_state".to_string(), optimizer_state);
    extra.insert(
        "dataloader_state".to_string(),
        serde_json::to_value(dataloader_state)?,
    );

    let metadata = CheckpointMetadata {
        step,
        loss,
        learning_rate: lr,
        extra,
    };

    // Save checkpoint using model's checkpoint functionality
    save_checkpoint(model, &checkpoint_path, Some(metadata))
        .with_context(|| format!("Failed to save checkpoint to {:?}", checkpoint_path))?;

    Ok(())
}

/// Resume training from a checkpoint
///
/// Loads model, optimizer state, and dataloader state from a checkpoint.
///
/// # Arguments
/// * `checkpoint_path` - Path to checkpoint file (without extension)
/// * `optimizer_config` - Optimizer configuration (used to recreate optimizer)
/// * `dataloader` - DataLoader to restore state to (mutated in place)
/// * `model` - Model to restore optimizer for (mutable reference needed)
///
/// # Returns
/// Optimizer, metadata, and training step (model is mutated in place)
///
/// # Note
/// Optimizer moment estimates (m, v) are not restored as aprender doesn't expose them.
/// The optimizer will restart with fresh moment estimates, which may cause a brief
/// adjustment period. Step count and learning rate are restored.
pub fn resume_from_checkpoint(
    checkpoint_path: &Path,
    optimizer_config: OptimizerConfig,
    dataloader: &mut DataLoader,
    model: &mut GPT,
) -> Result<(AdamW, CheckpointMetadata, usize)> {
    use std::ffi::OsStr;

    // Load checkpoint (returns model and metadata)
    let (loaded_model, metadata) = load_checkpoint(checkpoint_path)
        .with_context(|| format!("Failed to load checkpoint from {:?}", checkpoint_path))?;

    // Copy loaded model weights into the provided model
    // Note: This assumes model config matches. In a real scenario, we'd validate this.
    // For now, we'll use the loaded model directly by replacing the reference
    // Actually, we need to restructure - let's load into the provided model
    *model = loaded_model;

    // Get step from metadata, or extract from filename as fallback
    let step = if metadata.step > 0 {
        metadata.step
    } else {
        checkpoint_path
            .file_stem()
            .and_then(OsStr::to_str)
            .and_then(|s| s.strip_prefix("checkpoint_step_"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    };

    // Recreate optimizer (moment estimates will be fresh)
    let (mut optimizer, _scheduler) = setup_optimizers(model, optimizer_config.clone())
        .context("Failed to setup optimizers for resume")?;

    // Restore optimizer state (step count and LR) if available
    if let Some(opt_state_val) = metadata.extra.get("optimizer_state") {
        if let Some(lr) = opt_state_val.get("lr").and_then(|v| v.as_f64()) {
            optimizer.set_lr(lr as f32);
        }
        // TODO: Restore moment estimates if aprender exposes them
    } else if let Some(lr) = metadata.learning_rate {
        // Fallback: use LR from metadata
        optimizer.set_lr(lr);
    }

    // Restore dataloader state if available
    if let Some(state_val) = metadata.extra.get("dataloader_state") {
        let state: DataLoaderState = serde_json::from_value(state_val.clone())
            .context("Failed to parse dataloader state")?;
        dataloader.restore_state(state);
    }

    Ok((optimizer, metadata, step))
}
