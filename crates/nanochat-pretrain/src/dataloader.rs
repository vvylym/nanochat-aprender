//! Data loading for pretraining

use anyhow::{Context, Result};
use aprender::autograd::Tensor;
use nanochat_tokenizer::Tokenizer;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// DataLoader state for checkpointing
///
/// Stores the current position in the data stream and RNG seed
/// to allow resuming training from the exact same point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderState {
    /// Current position in token stream
    pub current_pos: usize,
    /// RNG seed (for reproducibility)
    pub rng_seed: u64,
}

/// DataLoader for pretraining
///
/// Loads text data, tokenizes it, and provides batches for training.
pub struct DataLoader {
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    batch_size: usize,
    seq_len: usize,
    #[allow(dead_code)]
    num_workers: usize,
    #[allow(dead_code)]
    token_ids: Vec<u32>,
    current_pos: usize,
    rng: StdRng,
    rng_seed: u64,
}

impl DataLoader {
    /// Create a new DataLoader
    ///
    /// # Arguments
    /// * `data_dir` - Directory containing text files
    /// * `tokenizer` - Tokenizer for encoding text
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `num_workers` - Number of worker threads (currently unused, for future use)
    pub fn new(
        data_dir: &Path,
        tokenizer: Tokenizer,
        batch_size: usize,
        seq_len: usize,
        num_workers: usize,
    ) -> Result<Self> {
        // Load and tokenize all text files in the directory
        let token_ids = Self::load_and_tokenize(data_dir, &tokenizer)
            .context("Failed to load and tokenize data")?;

        // Initialize RNG with a seed
        let rng_seed = 42; // TODO: Make configurable or use timestamp
        let rng = StdRng::seed_from_u64(rng_seed);

        Ok(Self {
            tokenizer,
            batch_size,
            seq_len,
            num_workers,
            token_ids,
            current_pos: 0,
            rng,
            rng_seed,
        })
    }

    /// Load text files and tokenize them
    fn load_and_tokenize(data_dir: &Path, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        let mut all_token_ids = Vec::new();

        // Read all .txt files in the directory
        let entries = fs::read_dir(data_dir).context("Failed to read data directory")?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                let text = fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read file: {:?}", path))?;

                let ids = tokenizer.encode(&text).context("Failed to tokenize text")?;

                all_token_ids.extend(ids);
            }
        }

        Ok(all_token_ids)
    }

    /// Get the next batch (inputs and targets)
    ///
    /// Returns a tuple of (inputs, targets) tensors, both of shape [batch_size, seq_len].
    /// Targets are inputs shifted by 1 position for next-token prediction.
    /// Returns None when all data has been consumed.
    ///
    /// # Python Reference
    /// Matches `dataloader.py:76-77`:
    /// ```python
    /// inputs_cpu = scratch[:-1]
    /// targets_cpu = scratch[1:]
    /// ```
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
                // Inputs: [0..seq_len], Targets: [1..seq_len+1] (shifted by 1)
                inputs_data.extend(seq[..self.seq_len].iter().map(|&id| id as f32));
                targets_data.extend(seq[1..].iter().map(|&id| id as f32));
            } else {
                // Extract inputs: [start, start+seq_len)
                inputs_data.extend(
                    self.token_ids[start..start + self.seq_len].iter().map(|&id| id as f32),
                );
                // Extract targets: [start+1, start+seq_len+1) - shifted by 1
                targets_data.extend(
                    self.token_ids[start + 1..start + self.seq_len + 1].iter().map(|&id| id as f32),
                );
            }

            self.current_pos = end;
        }

        // Create tensors
        let inputs = Tensor::new(&inputs_data, &[self.batch_size, self.seq_len]);
        let targets = Tensor::new(&targets_data, &[self.batch_size, self.seq_len]);

        Ok(Some((inputs, targets)))
    }

    /// Shuffle the token IDs
    fn shuffle(&mut self) {
        self.token_ids.shuffle(&mut self.rng);
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset the dataloader to the beginning
    pub fn reset(&mut self) {
        self.current_pos = 0;
        self.shuffle();
    }

    /// Get current state for checkpointing
    ///
    /// Returns the current position in the token stream and RNG seed
    /// so training can be resumed from the exact same point.
    ///
    /// # Returns
    /// DataLoaderState containing current position and RNG seed
    pub fn get_state(&self) -> DataLoaderState {
        DataLoaderState {
            current_pos: self.current_pos,
            rng_seed: self.rng_seed,
        }
    }

    /// Restore state from checkpoint
    ///
    /// Restores the data loader to a previous state, allowing training
    /// to resume from the exact same point in the data stream.
    ///
    /// # Arguments
    /// * `state` - DataLoaderState to restore
    pub fn restore_state(&mut self, state: DataLoaderState) {
        self.current_pos = state.current_pos;
        self.rng_seed = state.rng_seed;
        self.rng = StdRng::seed_from_u64(state.rng_seed);
    }
}

/// Data sharding support for distributed training
///
/// This is a placeholder for future distributed training support.
pub struct DataShard {
    shard_id: usize,
    num_shards: usize,
    #[allow(dead_code)]
    token_ids: Vec<u32>,
}

impl DataShard {
    /// Create a data shard from token IDs
    pub fn new(shard_id: usize, num_shards: usize, token_ids: Vec<u32>) -> Self {
        Self {
            shard_id,
            num_shards,
            token_ids,
        }
    }

    /// Get the shard ID
    pub fn shard_id(&self) -> usize {
        self.shard_id
    }

    /// Get the number of shards
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }
}
