//! Data loading for mid-training with conversational data

use anyhow::{Context, Result};
use aprender::autograd::Tensor;
use nanochat_tokenizer::Tokenizer;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Conversation message structure
#[derive(Debug, Clone, Deserialize)]
struct Message {
    role: String,
    content: String,
}

/// Conversation structure from JSONL
#[derive(Debug, Clone, Deserialize)]
struct Conversation {
    messages: Vec<Message>,
}

/// DataLoader state for checkpointing
///
/// Stores the current position in the data stream and RNG seed
/// to allow resuming training from the exact same point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderState {
    /// Current position in conversation list
    pub current_pos: usize,
    /// RNG seed (for reproducibility)
    pub rng_seed: u64,
}

/// DataLoader for mid-training with conversational data
///
/// Loads JSONL files containing conversations, converts them to token sequences,
/// and provides batches for training.
pub struct ConversationDataLoader {
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    batch_size: usize,
    seq_len: usize,
    #[allow(dead_code)]
    num_workers: usize,
    conversations: Vec<Conversation>,
    token_sequences: Vec<Vec<u32>>,
    current_pos: usize,
    rng: StdRng,
    rng_seed: u64,
}

impl ConversationDataLoader {
    /// Create a new ConversationDataLoader
    ///
    /// # Arguments
    /// * `data_dir` - Directory containing JSONL files with conversations
    /// * `tokenizer` - Tokenizer for encoding text
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `num_workers` - Number of worker threads (currently unused, for future use)
    /// * `seed` - Optional random seed for reproducibility (None = non-deterministic)
    pub fn new(
        data_dir: &Path,
        tokenizer: Tokenizer,
        batch_size: usize,
        seq_len: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<Self> {
        // Load conversations from JSONL files
        let conversations =
            Self::load_conversations(data_dir).context("Failed to load conversations")?;

        // Convert conversations to token sequences
        let token_sequences = Self::conversations_to_sequences(&conversations, &tokenizer)
            .context("Failed to convert conversations to token sequences")?;

        // Initialize RNG with provided seed or generate from entropy
        // Uses StdRng with SeedableRng::seed_from_u64() per Principle VII
        let rng_seed = seed.unwrap_or_else(|| {
            // Generate seed from current time if not provided
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_secs()
        });
        let rng = StdRng::seed_from_u64(rng_seed);

        Ok(Self {
            tokenizer,
            batch_size,
            seq_len,
            num_workers,
            conversations,
            token_sequences,
            current_pos: 0,
            rng,
            rng_seed,
        })
    }

    /// Load conversations from JSONL files in the directory
    fn load_conversations(data_dir: &Path) -> Result<Vec<Conversation>> {
        let mut conversations = Vec::new();

        let entries = fs::read_dir(data_dir).context("Failed to read data directory")?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            // Look for .jsonl files
            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                let file = fs::File::open(&path)
                    .with_context(|| format!("Failed to open file: {:?}", path))?;
                let reader = BufReader::new(file);

                for (line_num, line) in reader.lines().enumerate() {
                    let line = line.context("Failed to read line")?;
                    if line.trim().is_empty() {
                        continue;
                    }

                    let conversation: Conversation =
                        serde_json::from_str(&line).with_context(|| {
                            format!(
                                "Failed to parse conversation at line {} in {:?}",
                                line_num + 1,
                                path
                            )
                        })?;

                    // Validate conversation has at least one message
                    if conversation.messages.is_empty() {
                        continue;
                    }

                    conversations.push(conversation);
                }
            }
        }

        Ok(conversations)
    }

    /// Convert conversations to token sequences
    ///
    /// Formats conversations as: <|user|>content<|assistant|>content...
    /// This matches the format expected for conversational fine-tuning.
    fn conversations_to_sequences(
        conversations: &[Conversation],
        tokenizer: &Tokenizer,
    ) -> Result<Vec<Vec<u32>>> {
        let mut sequences = Vec::new();

        for conversation in conversations {
            // Format conversation as a single string
            let mut formatted = String::new();

            for message in &conversation.messages {
                // Format: <|role|>content
                // In practice, you might want to use special tokens from the tokenizer
                // For now, we'll use a simple format
                formatted.push_str(&format!("<|{}|>", message.role));
                formatted.push_str(&message.content);
            }

            // Tokenize the formatted conversation
            let tokens = tokenizer.encode(&formatted).context("Failed to tokenize conversation")?;

            if !tokens.is_empty() {
                sequences.push(tokens);
            }
        }

        Ok(sequences)
    }

    /// Get the next batch (inputs and targets)
    ///
    /// Returns a tuple of (inputs, targets) tensors, both of shape [batch_size, seq_len].
    /// Targets are inputs shifted by 1 position for next-token prediction.
    /// Returns None when all data has been consumed.
    pub fn next_batch(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        if self.token_sequences.is_empty() {
            return Ok(None);
        }

        // Check if we have enough sequences for a batch
        if self.current_pos >= self.token_sequences.len() {
            // Reset position and shuffle for next epoch
            self.current_pos = 0;
            self.shuffle();
        }

        // Extract sequences for the batch
        let mut inputs_data = Vec::new();
        let mut targets_data = Vec::new();

        for _ in 0..self.batch_size {
            if self.current_pos >= self.token_sequences.len() {
                // Not enough sequences, pad with zeros
                inputs_data.extend(vec![0.0; self.seq_len]);
                targets_data.extend(vec![0.0; self.seq_len]);
                continue;
            }

            let sequence = &self.token_sequences[self.current_pos];
            self.current_pos += 1;

            // Extract inputs and targets (shifted by 1)
            if sequence.len() > self.seq_len {
                // Enough tokens: inputs = [0..seq_len], targets = [1..seq_len+1]
                inputs_data.extend(sequence[..self.seq_len].iter().map(|&id| id as f32));
                targets_data.extend(sequence[1..=self.seq_len].iter().map(|&id| id as f32));
            } else {
                // Not enough tokens, pad with zeros
                let mut input_seq = sequence[..sequence.len().saturating_sub(1)]
                    .iter()
                    .map(|&id| id as f32)
                    .collect::<Vec<_>>();
                input_seq.resize(self.seq_len, 0.0);

                let mut target_seq = sequence[1..].iter().map(|&id| id as f32).collect::<Vec<_>>();
                target_seq.resize(self.seq_len, 0.0);

                inputs_data.extend(input_seq);
                targets_data.extend(target_seq);
            }
        }

        // Create tensors
        let inputs = Tensor::new(&inputs_data, &[self.batch_size, self.seq_len]);
        let targets = Tensor::new(&targets_data, &[self.batch_size, self.seq_len]);

        Ok(Some((inputs, targets)))
    }

    /// Shuffle the conversation sequences
    fn shuffle(&mut self) {
        // Shuffle both conversations and token_sequences together
        let mut indices: Vec<usize> = (0..self.conversations.len()).collect();
        indices.shuffle(&mut self.rng);

        let mut new_conversations = Vec::new();
        let mut new_sequences = Vec::new();

        for &idx in &indices {
            new_conversations.push(self.conversations[idx].clone());
            new_sequences.push(self.token_sequences[idx].clone());
        }

        self.conversations = new_conversations;
        self.token_sequences = new_sequences;
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get conversation count
    pub fn conversation_count(&self) -> usize {
        self.conversations.len()
    }

    /// Reset the dataloader to the beginning
    pub fn reset(&mut self) {
        self.current_pos = 0;
        self.shuffle();
    }

    /// Get current state for checkpointing
    ///
    /// Returns the current position in the conversation list and RNG seed
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
