//! Integration tests for training loop in pretraining

use aprender::autograd::Tensor;
use nanochat_model::{GPTConfig, GPT};
use nanochat_pretrain::train::train_step;

#[test]
fn test_training_step() {
    let model_config = GPTConfig::default();
    let mut model = GPT::new(model_config);

    // Create dummy input and targets
    let batch_size = 2;
    let seq_len = 10;
    let idx = Tensor::zeros(&[batch_size, seq_len]);
    let targets = Tensor::zeros(&[batch_size, seq_len]);

    // Test a single training step (forward + backward only, no optimizer step)
    let loss = train_step(&mut model, &idx, &targets).expect("Training step failed");

    // Loss should be a scalar tensor
    assert_eq!(loss.shape().len(), 1);
    assert!(loss.shape()[0] == 1);

    // Loss should be finite
    let loss_value = loss.item();
    assert!(loss_value.is_finite(), "Loss should be finite");
    assert!(loss_value >= 0.0, "Loss should be non-negative");
}

#[test]
fn test_gradient_accumulation() {
    let model_config = GPTConfig::default();
    let mut model = GPT::new(model_config);

    let batch_size = 2;
    let seq_len = 10;
    let idx = Tensor::zeros(&[batch_size, seq_len]);
    let targets = Tensor::zeros(&[batch_size, seq_len]);

    // Run multiple steps with gradient accumulation
    for _ in 0..4 {
        let loss = train_step(&mut model, &idx, &targets).expect("Training step failed");
        assert!(loss.item().is_finite());
    }
}

#[test]
fn test_forward_training_loss() {
    // Test that forward_training returns a loss tensor
    let model_config = GPTConfig::default();
    let model = GPT::new(model_config);

    let batch_size = 1;
    let seq_len = 5;
    let idx = Tensor::zeros(&[batch_size, seq_len]);
    let targets = Tensor::zeros(&[batch_size, seq_len]);

    // Use model's forward_training directly
    let loss = model.forward_training(&idx, &targets, None).expect("Forward training failed");

    // Loss should be a scalar
    assert_eq!(loss.shape().len(), 1);
    assert!(loss.shape()[0] == 1);

    // Loss should be finite and non-negative
    let loss_value = loss.item();
    assert!(loss_value.is_finite());
    assert!(loss_value >= 0.0);
}

#[test]
fn test_validation_evaluation() {
    use nanochat_pretrain::dataloader::DataLoader;
    use nanochat_pretrain::train::evaluate_validation_loss;
    use nanochat_tokenizer::Tokenizer;
    use std::fs;
    use tempfile::TempDir;

    // Create model
    let model_config = GPTConfig::default();
    let model = GPT::new(model_config);

    // Create temporary directory with validation data
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let val_file = temp_dir.path().join("val.txt");
    fs::write(&val_file, "hello world hello rust world peace ".repeat(10))
        .expect("Failed to write validation data");

    // Create tokenizer
    let corpus = ["hello world", "hello rust", "world peace"];
    let tokenizer =
        Tokenizer::train_from_iterator(corpus.iter(), 500).expect("Failed to train tokenizer");

    // Create validation dataloader
    let mut val_dataloader = DataLoader::new(
        temp_dir.path(),
        tokenizer,
        2, // batch_size
        5, // seq_len
        1, // num_workers
    )
    .expect("Failed to create validation dataloader");

    // Evaluate validation loss
    let val_bpb = evaluate_validation_loss(&model, &mut val_dataloader, 2)
        .expect("Validation evaluation failed");

    // Validation loss should be finite and non-negative
    assert!(val_bpb.is_finite());
    assert!(val_bpb >= 0.0);
}
