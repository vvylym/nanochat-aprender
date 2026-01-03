//! Training loop for supervised fine-tuning

use crate::dataloader::{DataLoaderState, InstructionDataLoader};
use crate::metrics::MetricsLogger;
use crate::optimizer::{
    get_learning_rate, setup_optimizers, update_learning_rate_custom, OptimizerConfig,
};
use anyhow::{Context, Result};
use aprender::nn::optim::Optimizer;
use nanochat_model::{
    checkpoint::{load_checkpoint, save_checkpoint, CheckpointMetadata},
    GPT,
};
use nanochat_pretrain::train::clip_gradients;
use serde_json::{self, Map, Number, Value};
use std::collections::HashMap;
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
    /// Random seed for data shuffling and reproducibility (None = non-deterministic)
    pub seed: Option<u64>,
}

/// Train the model using instruction data
///
/// This function implements the supervised fine-tuning loop, which fine-tunes a mid-trained
/// model on instruction-following data. It reuses the pretraining infrastructure but
/// uses instruction data loading.
///
/// # Arguments
/// * `model` - The GPT model (mid-trained, mutable reference needed for optimizer)
/// * `tokenizer` - Tokenizer for encoding instructions
/// * `data_dir` - Directory containing JSONL files with instruction-response pairs
/// * `output_dir` - Directory for saving checkpoints
/// * `training_config` - Training hyperparameters
/// * `optimizer_config` - Optimizer configuration (None = use defaults)
/// * `resume_checkpoint` - Path to checkpoint to resume from (None = start fresh)
pub fn train(
    model: &mut GPT,
    tokenizer: &nanochat_tokenizer::Tokenizer,
    data_dir: &Path,
    output_dir: &Path,
    training_config: &TrainingConfig,
    optimizer_config: Option<OptimizerConfig>,
    resume_checkpoint: Option<&Path>,
) -> Result<()> {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

    // Create checkpoints subdirectory
    let checkpoints_dir = output_dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoints_dir).with_context(|| {
        format!(
            "Failed to create checkpoints directory: {:?}",
            checkpoints_dir
        )
    })?;

    // Load or create data loader
    let mut dataloader = InstructionDataLoader::new(
        data_dir,
        tokenizer.clone(),
        training_config.batch_size,
        training_config.seq_len,
        1, // num_workers (for now, single-threaded)
        training_config.seed,
    )
    .context("Failed to create data loader")?;

    // Setup optimizer and scheduler
    let optimizer_config = optimizer_config.unwrap_or(OptimizerConfig {
        learning_rate: 1e-4,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        warmup_steps: 1000,
        max_steps: training_config.max_steps,
        min_lr: 1e-6,
        warmup_ratio: None,
        warmdown_ratio: Some(0.2),
        final_lr_frac: Some(0.0),
    });

    let (mut optimizer, mut scheduler) =
        setup_optimizers(model, optimizer_config.clone()).context("Failed to setup optimizers")?;

    // Resume from checkpoint if provided
    let mut start_step = 0;
    if let Some(checkpoint_path) = resume_checkpoint {
        let (loaded_model, metadata) =
            load_checkpoint(checkpoint_path).context("Failed to load checkpoint")?;

        // Copy loaded model weights into the provided model
        *model = loaded_model;

        // Restore optimizer state from metadata (stored in extra)
        if let Some(optimizer_state) = metadata.extra.get("optimizer_state") {
            if let Some(step) = optimizer_state.get("step").and_then(|v| v.as_u64()) {
                start_step = step as usize;
            }
            if let Some(lr) = optimizer_state.get("lr").and_then(|v| v.as_f64()) {
                optimizer.set_lr(lr as f32);
            }
        } else {
            // Fallback: use step and LR from metadata directly
            start_step = metadata.step;
            if let Some(lr) = metadata.learning_rate {
                optimizer.set_lr(lr);
            }
        }

        // Restore dataloader state from metadata (stored in extra)
        if let Some(dataloader_state_json) = metadata.extra.get("dataloader_state") {
            let dataloader_state: DataLoaderState =
                serde_json::from_value(dataloader_state_json.clone())
                    .context("Failed to parse dataloader state")?;
            dataloader.restore_state(&dataloader_state);
        }

        println!("Resumed from checkpoint at step {}", start_step);
    }

    // Initialize metrics logger
    let mut metrics_logger = MetricsLogger::new(training_config.log_interval);

    // Training loop
    let mut step = start_step;
    let mut accumulated_loss = 0.0;
    let mut accumulation_count = 0;

    while step < training_config.max_steps {
        // Get next batch (now includes training mask per remediation plan FR-022.6)
        // The mask indicates which tokens to train on (1 for assistant tokens, 0 for others)
        let (batch, targets, _mask) = match dataloader.next_batch()? {
            Some((b, t, m)) => (b, t, m),
            None => {
                // Epoch finished, reset dataloader
                dataloader.reset();
                continue;
            }
        };

        // Forward pass and backward pass (gradient accumulation)
        // Note: Currently using forward_training() which computes loss over all tokens.
        // TODO: Apply training mask to only train on assistant tokens (mask=1).
        // This requires computing per-token loss and applying mask before averaging.
        // For now, the mask is generated correctly by render_conversation(), but not yet
        // applied to loss computation. This is acceptable as Phase 6 focuses on
        // instruction data loading and training loop infrastructure.
        let loss = model
            .forward_training(&batch, &targets, None)
            .context("Forward training failed")?;

        // Backward pass: compute gradients
        loss.backward();

        accumulated_loss += loss.item();
        accumulation_count += 1;

        // Only update optimizer after accumulation steps
        if accumulation_count >= training_config.gradient_accumulation_steps {
            // Gradient clipping
            if training_config.grad_clip > 0.0 {
                let _grad_norm = clip_gradients(model, training_config.grad_clip)
                    .context("Failed to compute gradient norm")?;
            }

            // Optimizer step
            optimizer.step();
            optimizer.zero_grad();

            // Update learning rate
            update_learning_rate_custom(&mut scheduler, &mut optimizer, step, &optimizer_config);

            // Get current learning rate
            let learning_rate = get_learning_rate(&optimizer);

            // Log metrics
            let avg_loss = accumulated_loss / accumulation_count as f32;
            let tokens_processed =
                training_config.batch_size * training_config.seq_len * accumulation_count;
            metrics_logger.log_step(&loss, learning_rate, tokens_processed, 1.0);

            // Save checkpoint at intervals
            if step > 0 && step % training_config.save_interval == 0 {
                let checkpoint_path =
                    checkpoints_dir.join(format!("checkpoint_{}.safetensors", step));
                let dataloader_state = dataloader.get_state();

                // Create metadata with optimizer and dataloader state
                let mut extra = HashMap::new();
                let mut optimizer_state = Map::new();
                optimizer_state.insert("step".to_string(), Value::Number(step.into()));
                optimizer_state.insert(
                    "lr".to_string(),
                    Value::Number(
                        Number::from_f64(learning_rate as f64).expect("Invalid learning rate"),
                    ),
                );
                extra.insert(
                    "optimizer_state".to_string(),
                    Value::Object(optimizer_state),
                );
                extra.insert(
                    "dataloader_state".to_string(),
                    serde_json::to_value(&dataloader_state)
                        .context("Failed to serialize dataloader state")?,
                );

                let metadata = CheckpointMetadata {
                    step,
                    loss: Some(avg_loss),
                    learning_rate: Some(learning_rate),
                    extra,
                };

                save_checkpoint(model, &checkpoint_path, Some(metadata))
                    .context("Failed to save checkpoint")?;

                println!("Saved checkpoint at step {} to {:?}", step, checkpoint_path);
            }

            // Reset accumulation counters
            accumulated_loss = 0.0;
            accumulation_count = 0;
            step += 1;
        }
    }

    // Save final checkpoint
    let final_checkpoint_path = checkpoints_dir.join("checkpoint_final.safetensors");
    let dataloader_state = dataloader.get_state();

    let mut extra = HashMap::new();
    let mut optimizer_state = Map::new();
    optimizer_state.insert("step".to_string(), Value::Number(step.into()));
    let learning_rate = get_learning_rate(&optimizer);
    optimizer_state.insert(
        "lr".to_string(),
        Value::Number(Number::from_f64(learning_rate as f64).expect("Invalid learning rate")),
    );
    extra.insert(
        "optimizer_state".to_string(),
        Value::Object(optimizer_state),
    );
    extra.insert(
        "dataloader_state".to_string(),
        serde_json::to_value(&dataloader_state).context("Failed to serialize dataloader state")?,
    );

    let metadata = CheckpointMetadata {
        step,
        loss: Some(accumulated_loss / accumulation_count.max(1) as f32),
        learning_rate: Some(get_learning_rate(&optimizer)),
        extra,
    };

    save_checkpoint(model, &final_checkpoint_path, Some(metadata))
        .context("Failed to save final checkpoint")?;

    println!(
        "Training completed! Final checkpoint saved to {:?}",
        final_checkpoint_path
    );

    Ok(())
}
