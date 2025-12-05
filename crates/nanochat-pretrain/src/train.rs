//! Training loop for pretraining

use crate::dataloader::DataLoader;
use crate::metrics::MetricsLogger;
use crate::optimizer::{
    get_learning_rate, setup_optimizers, update_learning_rate, OptimizerConfig,
};
use anyhow::{Context, Result};
use aprender::autograd::Tensor;
use aprender::nn::optim::Optimizer;
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
fn clip_gradients(_model: &mut GPT, max_norm: f32) -> Result<f32> {
    if max_norm <= 0.0 {
        return Ok(0.0); // Clipping disabled
    }

    // TODO: Implement gradient clipping once aprender's gradient access API is understood
    // Gradient clipping requires:
    // 1. Access to gradient values from parameters (not parameter values)
    // 2. Computing L2 norm across all gradients
    // 3. Scaling gradients by clip_coef if norm exceeds max_norm
    //
    // In PyTorch: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    // In aprender: Need to find equivalent API for accessing gradients
    //
    // For now, we return 0.0 to indicate no clipping was performed
    // This allows the training loop to compile and run, but gradients won't be clipped
    eprintln!(
        "Warning: Gradient clipping requested (max_norm={}) but not yet implemented. \
         Need to determine aprender's gradient access API.",
        max_norm
    );

    Ok(0.0)
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
    let (mut optimizer, mut scheduler) =
        setup_optimizers(&mut model, optimizer_config).context("Failed to setup optimizers")?;

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
            if training_config.grad_clip > 0.0 {
                clip_gradients(&mut model, training_config.grad_clip)
                    .context("Failed to clip gradients")?;
            }

            // Optimizer step: update parameters using accumulated gradients
            optimizer.step();

            // Zero gradients for next iteration
            optimizer.zero_grad();

            // Update learning rate using scheduler
            update_learning_rate(&mut scheduler, &mut optimizer);

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
        update_learning_rate(&mut scheduler, &mut optimizer);
        let learning_rate = get_learning_rate(&optimizer);
        save_checkpoint_step(
            &model,
            output_dir,
            step,
            Some(avg_loss),
            Some(learning_rate),
        )
        .context("Failed to save final checkpoint")?;
    } else {
        let learning_rate = get_learning_rate(&optimizer);
        save_checkpoint_step(&model, output_dir, step, None, Some(learning_rate))
            .context("Failed to save final checkpoint")?;
    }

    Ok(())
}

/// Save a training checkpoint
///
/// # Arguments
/// * `model` - The GPT model
/// * `output_dir` - Directory to save checkpoint
/// * `step` - Training step number
/// * `loss` - Optional loss value
/// * `lr` - Optional learning rate
fn save_checkpoint_step(
    model: &GPT,
    output_dir: &Path,
    step: usize,
    loss: Option<f32>,
    lr: Option<f32>,
) -> Result<()> {
    // Create checkpoint path (without extension, save_checkpoint will add .safetensors and .json)
    let checkpoint_path = output_dir.join(format!("checkpoint_step_{}", step));

    // Create metadata
    let metadata = CheckpointMetadata {
        step,
        loss,
        learning_rate: lr,
        extra: std::collections::HashMap::new(),
    };

    // Save checkpoint using model's checkpoint functionality
    save_checkpoint(model, &checkpoint_path, Some(metadata))
        .with_context(|| format!("Failed to save checkpoint to {:?}", checkpoint_path))?;

    Ok(())
}

/// Resume training from a checkpoint
///
/// # Arguments
/// * `checkpoint_path` - Path to checkpoint file (without extension)
///
/// # Returns
/// Loaded model, metadata, and training step
pub fn resume_from_checkpoint(checkpoint_path: &Path) -> Result<(GPT, CheckpointMetadata, usize)> {
    use std::ffi::OsStr;

    // Load checkpoint (returns model and metadata)
    let (model, metadata) = load_checkpoint(checkpoint_path)
        .with_context(|| format!("Failed to load checkpoint from {:?}", checkpoint_path))?;

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

    Ok((model, metadata, step))
}
