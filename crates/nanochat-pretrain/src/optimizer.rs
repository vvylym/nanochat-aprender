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

    // Create learning rate scheduler
    // WarmupCosineScheduler::new takes (warmup_steps, total_steps)
    let scheduler = WarmupCosineScheduler::new(config.warmup_steps, config.max_steps);

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
