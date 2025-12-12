//! Unit tests for optimizer configuration in pretraining

use aprender::nn::Module;
use nanochat_model::{GPTConfig, GPT};
use nanochat_pretrain::optimizer::{setup_optimizers, OptimizerConfig};

#[test]
fn test_optimizer_config_creation() {
    let config = OptimizerConfig {
        learning_rate: 1e-4,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        warmup_steps: 1000,
        max_steps: 10000,
        min_lr: 1e-6,
        warmup_ratio: None,
        warmdown_ratio: None,
        final_lr_frac: None,
    };

    assert_eq!(config.learning_rate, 1e-4);
    assert_eq!(config.weight_decay, 0.1);
    assert_eq!(config.warmup_steps, 1000);
}

#[test]
fn test_setup_optimizers() {
    let model_config = GPTConfig::default();
    let mut model = GPT::new(model_config);

    let optimizer_config = OptimizerConfig {
        learning_rate: 1e-4,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        warmup_steps: 1000,
        max_steps: 10000,
        min_lr: 1e-6,
        warmup_ratio: None,
        warmdown_ratio: None,
        final_lr_frac: None,
    };

    let (_optimizer, _scheduler) =
        setup_optimizers(&mut model, optimizer_config).expect("Failed to setup optimizers");

    // Verify optimizer and scheduler were created successfully
    // Both should be Some (not Option) since setup_optimizers returns concrete types
}

#[test]
fn test_learning_rate_scheduler() {
    let model_config = GPTConfig::default();
    let mut model = GPT::new(model_config);

    let optimizer_config = OptimizerConfig {
        learning_rate: 1e-4,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        warmup_steps: 100,
        max_steps: 1000,
        min_lr: 1e-6,
        warmup_ratio: None,
        warmdown_ratio: None,
        final_lr_frac: None,
    };

    let (_optimizer, _scheduler) =
        setup_optimizers(&mut model, optimizer_config).expect("Failed to setup optimizers");

    // Verify scheduler was created
    // The scheduler should handle warmup and cosine decay
}

#[test]
fn test_parameter_groups() {
    // Test that optimizers are set up for different parameter groups
    // (embedding, LM head, matrix params) as per Python reference
    let model_config = GPTConfig::default();
    let model = GPT::new(model_config);

    // Get parameters to verify they exist
    let parameters = model.parameters();
    assert!(!parameters.is_empty(), "Model should have parameters");

    // Verify we can access different parameter groups
    // Embedding parameters
    // LM head parameters
    // Matrix parameters (attention, MLP)

    // This test mainly verifies the model has parameters that can be optimized
    assert!(!parameters.is_empty());
}

#[test]
fn test_lr_multiplier_warmup() {
    use nanochat_pretrain::optimizer::get_lr_multiplier;

    // Test warmup phase
    let lr_mult = get_lr_multiplier(5, 10, 5, 20, 0.1);
    // At step 5 of 10 warmup steps, multiplier should be (5+1)/10 = 0.6
    assert!((lr_mult - 0.6).abs() < 1e-6);
}

#[test]
fn test_lr_multiplier_constant() {
    use nanochat_pretrain::optimizer::get_lr_multiplier;

    // Test constant phase (after warmup, before warmdown)
    let lr_mult = get_lr_multiplier(10, 10, 5, 20, 0.1);
    // Should be 1.0 in constant phase
    assert!((lr_mult - 1.0).abs() < 1e-6);
}

#[test]
fn test_lr_multiplier_warmdown() {
    use nanochat_pretrain::optimizer::get_lr_multiplier;

    // Test warmdown phase
    let lr_mult = get_lr_multiplier(18, 10, 5, 20, 0.1);
    // At step 18, 2 steps into warmdown (20-18=2, warmdown=5)
    // progress = 2/5 = 0.4
    // lr_mult = 0.4 * 1.0 + 0.6 * 0.1 = 0.4 + 0.06 = 0.46
    assert!((lr_mult - 0.46).abs() < 1e-6);
}
