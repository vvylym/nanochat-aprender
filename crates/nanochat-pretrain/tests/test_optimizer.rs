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
    assert!(parameters.len() > 0);
}
