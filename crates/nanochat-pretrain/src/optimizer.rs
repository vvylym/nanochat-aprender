//! Optimizer and learning rate scheduler setup for pretraining

use anyhow::Result;
use aprender::nn::{
    optim::{AdamW, Optimizer},
    scheduler::{LRScheduler, WarmupCosineScheduler},
    Module,
};
use nanochat_model::GPT;

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Beta1 for AdamW
    pub beta1: f32,
    /// Beta2 for AdamW
    pub beta2: f32,
    /// Epsilon for AdamW
    pub eps: f32,
    /// Number of warmup steps
    pub warmup_steps: usize,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Warmup ratio (alternative to warmup_steps, calculated as ratio of max_steps)
    pub warmup_ratio: Option<f32>,
    /// Warmdown ratio (for cosine decay, calculated as ratio of max_steps)
    pub warmdown_ratio: Option<f32>,
    /// Final LR fraction (for cosine decay, final LR = initial LR * final_lr_frac)
    pub final_lr_frac: Option<f32>,
}

/// Setup optimizers for training
///
/// This implements the `setup_optimizers()` functionality from Python reference.
/// It configures optimizers for model parameters using aprender's AdamW.
///
/// Note: In Rust, we need to work with the model's parameters_mut() method
/// to get mutable references. The optimizer will be created when we have
/// mutable access to the model.
///
/// # Arguments
/// * `model` - The GPT model (mutable reference needed for parameters)
/// * `config` - Optimizer configuration
///
/// # Returns
/// Tuple of (optimizer, scheduler)
///
/// # Note
/// If `warmup_ratio` is provided, it overrides `warmup_steps`.
/// aprender's WarmupCosineScheduler uses cosine decay, which may differ from
/// Python's linear warmdown. For exact Python behavior, use custom LR scheduling.
pub fn setup_optimizers(
    model: &mut GPT,
    config: OptimizerConfig,
) -> Result<(AdamW, WarmupCosineScheduler)> {
    // Get mutable references to all model parameters
    // This uses aprender's Module trait which provides parameters_mut()
    let parameters = model.parameters_mut();

    if parameters.is_empty() {
        anyhow::bail!("Model has no parameters to optimize");
    }

    // Create AdamW optimizer with all parameters
    // aprender's AdamW::new takes Vec<&mut Tensor> and learning rate
    let optimizer = AdamW::new(parameters, config.learning_rate);

    // Determine warmup steps (use ratio if provided, otherwise use warmup_steps)
    let warmup_steps = if let Some(ratio) = config.warmup_ratio {
        (ratio * config.max_steps as f32) as usize
    } else {
        config.warmup_steps
    };

    // Create learning rate scheduler
    // WarmupCosineScheduler supports min_lr via with_min_lr
    // If final_lr_frac is provided, use it to calculate min_lr
    let scheduler = if let Some(final_lr_frac) = config.final_lr_frac {
        let min_lr = config.learning_rate * final_lr_frac;
        WarmupCosineScheduler::with_min_lr(warmup_steps, config.max_steps, min_lr)
    } else {
        WarmupCosineScheduler::with_min_lr(warmup_steps, config.max_steps, config.min_lr)
    };

    Ok((optimizer, scheduler))
}

/// Update learning rate using scheduler
///
/// # Arguments
/// * `scheduler` - The learning rate scheduler
/// * `optimizer` - The optimizer to update
///
/// This calls the scheduler's step method which updates the optimizer's learning rate.
pub fn update_learning_rate(scheduler: &mut WarmupCosineScheduler, optimizer: &mut AdamW) {
    // aprender's scheduler step() method takes an optimizer reference
    // and updates the optimizer's learning rate internally
    scheduler.step(optimizer);
}

/// Get current learning rate from optimizer
///
/// # Arguments
/// * `optimizer` - The optimizer
///
/// # Returns
/// Current learning rate
pub fn get_learning_rate(optimizer: &AdamW) -> f32 {
    optimizer.lr()
}

/// Learning rate multiplier function (matching Python's get_lr_multiplier)
///
/// Computes the learning rate multiplier for a given step, supporting
/// warmup, constant, and warmdown phases.
///
/// # Arguments
/// * `step` - Current training step (0-indexed)
/// * `warmup_steps` - Number of warmup steps
/// * `warmdown_steps` - Number of warmdown steps
/// * `total_steps` - Total number of training steps
/// * `final_lr_frac` - Final learning rate as fraction of initial LR
///
/// # Returns
/// Learning rate multiplier (0.0 to 1.0)
///
/// # Python Reference
/// Matches `base_train.py:180-189`:
/// ```python
/// def get_lr_multiplier(step, warmup_steps, warmdown_steps, total_steps, final_lr_frac):
///     if step < warmup_steps:
///         return (step + 1) / warmup_steps
///     elif step <= total_steps - warmdown_steps:
///         return 1.0
///     else:
///         progress = (total_steps - step) / warmdown_steps
///         return progress * 1.0 + (1.0 - progress) * final_lr_frac
/// ```
pub fn get_lr_multiplier(
    step: usize,
    warmup_steps: usize,
    warmdown_steps: usize,
    total_steps: usize,
    final_lr_frac: f32,
) -> f32 {
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

/// Update learning rate using custom scheduler (if warmdown_ratio is specified)
///
/// If `warmdown_ratio` and `final_lr_frac` are provided in config, this function
/// implements Python's linear warmdown behavior. Otherwise, it uses aprender's
/// WarmupCosineScheduler which uses cosine decay.
///
/// # Arguments
/// * `scheduler` - The learning rate scheduler (may be unused if custom logic is used)
/// * `optimizer` - The optimizer to update
/// * `step` - Current training step
/// * `config` - Optimizer configuration
///
/// # Note
/// aprender's WarmupCosineScheduler uses cosine decay, which differs from Python's
/// linear warmdown. This function provides a workaround for exact Python behavior
/// when warmdown_ratio and final_lr_frac are specified.
pub fn update_learning_rate_custom(
    scheduler: &mut WarmupCosineScheduler,
    optimizer: &mut AdamW,
    step: usize,
    config: &OptimizerConfig,
) {
    // Use custom LR scheduling if warmdown_ratio and final_lr_frac are provided
    // This matches Python's linear warmdown behavior
    if let (Some(warmdown_ratio), Some(final_lr_frac)) =
        (config.warmdown_ratio, config.final_lr_frac)
    {
        // Calculate warmup and warmdown steps
        let warmup_steps = if let Some(ratio) = config.warmup_ratio {
            (ratio * config.max_steps as f32) as usize
        } else {
            config.warmup_steps
        };
        let warmdown_steps = (warmdown_ratio * config.max_steps as f32) as usize;

        // Compute LR multiplier using Python's algorithm
        let lr_mult = get_lr_multiplier(
            step,
            warmup_steps,
            warmdown_steps,
            config.max_steps,
            final_lr_frac,
        );

        // Manually set learning rate
        optimizer.set_lr(config.learning_rate * lr_mult);
    } else {
        // Use aprender's scheduler (cosine decay)
        scheduler.step(optimizer);
    }
}
