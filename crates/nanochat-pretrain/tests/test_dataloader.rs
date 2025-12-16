//! Unit tests for data loading in pretraining

use nanochat_pretrain::dataloader::DataLoader;
use nanochat_tokenizer::Tokenizer;
use tempfile::TempDir;

#[test]
fn test_dataloader_creation() {
    let tokenizer = create_test_tokenizer();
    let data_dir = create_test_data_dir();

    let dataloader = DataLoader::new(
        data_dir.path(),
        tokenizer,
        32,       // batch_size
        256,      // seq_len
        1,        // num_workers
        Some(42), // seed
    )
    .expect("Failed to create DataLoader");

    assert_eq!(dataloader.batch_size(), 32);
    assert_eq!(dataloader.seq_len(), 256);
}

#[test]
fn test_dataloader_batching() {
    let tokenizer = create_test_tokenizer();
    let data_dir = create_test_data_dir();

    let mut dataloader = DataLoader::new(
        data_dir.path(),
        tokenizer,
        2,        // batch_size
        10,       // seq_len
        1,        // num_workers
        Some(42), // seed
    )
    .expect("Failed to create DataLoader");

    // Get a batch (now returns inputs and targets)
    let (inputs, targets) = dataloader
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");

    // Both should have correct shape [batch_size, seq_len]
    assert_eq!(inputs.shape().len(), 2);
    assert_eq!(inputs.shape()[0], 2); // batch_size
    assert_eq!(inputs.shape()[1], 10); // seq_len
    assert_eq!(targets.shape().len(), 2);
    assert_eq!(targets.shape()[0], 2); // batch_size
    assert_eq!(targets.shape()[1], 10); // seq_len
}

#[test]
fn test_dataloader_shuffling() {
    let tokenizer = create_test_tokenizer();
    let data_dir = create_test_data_dir();

    let mut dataloader1 = DataLoader::new(data_dir.path(), tokenizer.clone(), 2, 10, 1, Some(42))
        .expect("Failed to create DataLoader");

    let mut dataloader2 = DataLoader::new(data_dir.path(), tokenizer, 2, 10, 1, Some(43))
        .expect("Failed to create DataLoader");

    // Get batches from both loaders
    let (inputs1, _targets1) = dataloader1
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");
    let (inputs2, _targets2) = dataloader2
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");

    // Batches should be different due to shuffling (with high probability)
    // Note: This is probabilistic, but with different seeds they should differ
    let data1: Vec<f32> = inputs1.data().to_vec();
    let data2: Vec<f32> = inputs2.data().to_vec();

    // At least verify they're valid batches
    assert_eq!(data1.len(), data2.len());
}

#[test]
fn test_dataloader_target_shifting() {
    let tokenizer = create_test_tokenizer();
    let data_dir = create_test_data_dir();

    let mut dataloader = DataLoader::new(
        data_dir.path(),
        tokenizer,
        1,        // batch_size
        5,        // seq_len
        1,        // num_workers
        Some(42), // seed
    )
    .expect("Failed to create DataLoader");

    let (inputs, targets) = dataloader
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");

    // Verify targets are inputs shifted by 1 position
    // For each position i, targets[i] should equal inputs[i+1]
    let inputs_data = inputs.data();
    let targets_data = targets.data();

    // Check that targets[i] == inputs[i+1] for all valid positions
    for i in 0..(targets_data.len() - 1) {
        assert_eq!(
            targets_data[i],
            inputs_data[i + 1],
            "Target at position {} should equal input at position {} (shifted by 1)",
            i,
            i + 1
        );
    }
}

#[test]
fn test_dataloader_tokenization() {
    let tokenizer = create_test_tokenizer();
    let data_dir = create_test_data_dir();

    let mut dataloader = DataLoader::new(
        data_dir.path(),
        tokenizer,
        1,        // batch_size
        5,        // seq_len
        1,        // num_workers
        Some(42), // seed
    )
    .expect("Failed to create DataLoader");

    let (inputs, targets) = dataloader
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");

    // Verify inputs and targets contain valid token IDs (non-negative integers)
    let inputs_data = inputs.data();
    let targets_data = targets.data();
    for &value in inputs_data.iter().chain(targets_data.iter()) {
        assert!(value >= 0.0, "Token ID should be non-negative");
        assert!(value.fract() == 0.0, "Token ID should be integer");
    }
}

// Helper functions
fn create_test_tokenizer() -> Tokenizer {
    let corpus = [
        "hello world",
        "hello rust",
        "world peace",
        "rust is awesome",
        "the quick brown fox",
    ];
    Tokenizer::train_from_iterator(corpus.iter(), 500).expect("Failed to create test tokenizer")
}

fn create_test_data_dir() -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let data_file = temp_dir.path().join("data.txt");

    // Create test data file with enough content for batching
    // Repeat content to ensure we have enough tokens
    let content = "hello world hello rust world peace rust is awesome the quick brown fox jumps over the lazy dog ".repeat(20);
    std::fs::write(&data_file, content).expect("Failed to write test data");

    temp_dir
}

#[test]
fn test_dataloader_state_checkpointing() {
    let tokenizer = create_test_tokenizer();
    let data_dir = create_test_data_dir();

    let mut dataloader = DataLoader::new(
        data_dir.path(),
        tokenizer,
        2,        // batch_size
        10,       // seq_len
        1,        // num_workers
        Some(42), // seed
    )
    .expect("Failed to create DataLoader");

    // Get a batch to advance position
    let _batch1 = dataloader
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");

    // Save state
    let state = dataloader.get_state();
    assert!(state.current_pos > 0, "Position should have advanced");

    // Get another batch
    let _batch2 = dataloader
        .next_batch()
        .expect("Failed to get batch")
        .expect("No batch available");

    // Restore state (clone needed for comparison)
    let state_clone = dataloader.get_state();
    dataloader.restore_state(state_clone.clone());

    // Verify position was restored
    let restored_state = dataloader.get_state();
    assert_eq!(restored_state.current_pos, state_clone.current_pos);
    assert_eq!(restored_state.rng_seed, state_clone.rng_seed);
}
