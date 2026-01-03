//! Data loading for supervised fine-tuning with instruction data

use anyhow::{Context, Result};
use aprender::autograd::Tensor;
use nanochat_tokenizer::{Conversation, Message, MessageContent, Tokenizer};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Instruction-response pair structure from JSONL
#[derive(Debug, Clone, Deserialize)]
struct InstructionPair {
    instruction: String,
    response: String,
}

/// DataLoader state for checkpointing
///
/// Stores the current position in the data stream and RNG seed
/// to allow resuming training from the exact same point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderState {
    /// Current position in instruction list
    pub current_pos: usize,
    /// RNG seed (for reproducibility)
    pub rng_seed: u64,
}

/// DataLoader for supervised fine-tuning with instruction data
///
/// Loads JSONL files containing instruction-response pairs, converts them to conversations,
/// and provides batches for training. Uses the same render_conversation() method as mid-training
/// to ensure consistent tokenization and training mask generation.
pub struct InstructionDataLoader {
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    batch_size: usize,
    seq_len: usize,
    #[allow(dead_code)]
    num_workers: usize,
    conversations: Vec<Conversation>,
    tokenized_conversations: Vec<(Vec<u32>, Vec<u8>)>, // (token_ids, training_mask)
    current_pos: usize,
    rng: StdRng,
    rng_seed: u64,
}

impl InstructionDataLoader {
    /// Create a new InstructionDataLoader
    ///
    /// # Arguments
    /// * `data_dir` - Directory containing JSONL files with instruction-response pairs
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
        // Load instruction-response pairs from JSONL files
        let instruction_pairs =
            Self::load_instructions(data_dir).context("Failed to load instructions")?;

        // Convert instruction-response pairs to conversations
        // Format: instruction as user message, response as assistant message
        let conversations: Vec<Conversation> = instruction_pairs
            .into_iter()
            .map(|pair| Conversation {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: MessageContent::String(pair.instruction),
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: MessageContent::String(pair.response),
                    },
                ],
            })
            .collect();

        // Tokenize conversations using render_conversation() (reuse mid-training logic)
        let mut tokenized_conversations = Vec::new();
        for conversation in &conversations {
            let (ids, mask) = tokenizer
                .render_conversation(conversation, seq_len)
                .context("Failed to tokenize conversation with render_conversation()")?;
            tokenized_conversations.push((ids, mask));
        }

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
        let mut rng = StdRng::seed_from_u64(rng_seed);

        // Shuffle conversations
        let mut indices: Vec<usize> = (0..conversations.len()).collect();
        indices.shuffle(&mut rng);

        // Reorder conversations and tokenized data according to shuffled indices
        let shuffled_conversations: Vec<Conversation> =
            indices.iter().map(|&i| conversations[i].clone()).collect();
        let shuffled_tokenized: Vec<(Vec<u32>, Vec<u8>)> =
            indices.iter().map(|&i| tokenized_conversations[i].clone()).collect();

        Ok(Self {
            tokenizer,
            batch_size,
            seq_len,
            num_workers,
            conversations: shuffled_conversations,
            tokenized_conversations: shuffled_tokenized,
            current_pos: 0,
            rng,
            rng_seed,
        })
    }

    /// Load instruction-response pairs from JSONL files
    ///
    /// # Arguments
    /// * `data_dir` - Directory containing .jsonl files
    ///
    /// # Returns
    /// Vector of instruction-response pairs
    fn load_instructions(data_dir: &Path) -> Result<Vec<InstructionPair>> {
        let mut pairs = Vec::new();
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

                    let pair: InstructionPair = serde_json::from_str(&line).with_context(|| {
                        format!(
                            "Failed to parse instruction pair at line {} in {:?}",
                            line_num + 1,
                            path
                        )
                    })?;

                    // Validate pair has both fields
                    if pair.instruction.trim().is_empty() || pair.response.trim().is_empty() {
                        continue;
                    }

                    pairs.push(pair);
                }
            }
        }

        Ok(pairs)
    }

    /// Get the next batch (inputs, targets, and training mask)
    ///
    /// Returns a tuple of (inputs, targets, mask) tensors:
    /// - inputs: [batch_size, seq_len] - input token IDs
    /// - targets: [batch_size, seq_len] - target token IDs (shifted by 1)
    /// - mask: [batch_size, seq_len] - training mask (1.0 for assistant tokens, 0.0 for others)
    ///
    /// Returns None when all conversations have been consumed.
    ///
    /// # Remediation Plan Compliance
    /// Per FR-022.6 and T074.1, uses render_conversation() for proper tokenization
    /// and generates training mask where mask=1 for assistant tokens to train on.
    pub fn next_batch(&mut self) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        if self.tokenized_conversations.is_empty() {
            return Ok(None);
        }

        // Check if we have enough conversations for a batch
        if self.current_pos >= self.tokenized_conversations.len() {
            // Reset and reshuffle for next epoch
            self.current_pos = 0;
            self.shuffle();
            if self.current_pos >= self.tokenized_conversations.len() {
                return Ok(None);
            }
        }

        let mut inputs_data = Vec::new();
        let mut targets_data = Vec::new();
        let mut mask_data = Vec::new();

        for _ in 0..self.batch_size {
            if self.current_pos >= self.tokenized_conversations.len() {
                // Pad with zeros if we run out of conversations
                inputs_data.extend(std::iter::repeat_n(0.0, self.seq_len));
                targets_data.extend(std::iter::repeat_n(0.0, self.seq_len));
                mask_data.extend(std::iter::repeat_n(0.0, self.seq_len));
                continue;
            }

            let (ids, mask) = &self.tokenized_conversations[self.current_pos];
            self.current_pos += 1;

            // Extract sequence (truncate or pad to seq_len)
            let seq_len_actual = ids.len().min(self.seq_len);
            let ids_slice = &ids[..seq_len_actual];
            let mask_slice = &mask[..seq_len_actual];

            // Inputs: [0..seq_len-1]
            // Targets: [1..seq_len] (shifted by 1)
            // Mask: [0..seq_len-1] (for inputs)
            if seq_len_actual > 0 {
                // Inputs
                if seq_len_actual > 1 {
                    inputs_data.extend(ids_slice[..seq_len_actual - 1].iter().map(|&id| id as f32));
                }
                // Pad if needed
                let pad_len = self.seq_len.saturating_sub(seq_len_actual.saturating_sub(1));
                inputs_data.extend(std::iter::repeat_n(0.0, pad_len));

                // Targets (shifted by 1)
                if seq_len_actual > 1 {
                    targets_data.extend(ids_slice[1..seq_len_actual].iter().map(|&id| id as f32));
                }
                // Pad if needed
                targets_data.extend(std::iter::repeat_n(0.0, pad_len));

                // Mask (for inputs, same length as inputs)
                if seq_len_actual > 1 {
                    mask_data.extend(mask_slice[..seq_len_actual - 1].iter().map(|&m| m as f32));
                }
                // Pad if needed
                mask_data.extend(std::iter::repeat_n(0.0, pad_len));
            } else {
                // Empty conversation - pad with zeros
                inputs_data.extend(std::iter::repeat_n(0.0, self.seq_len));
                targets_data.extend(std::iter::repeat_n(0.0, self.seq_len));
                mask_data.extend(std::iter::repeat_n(0.0, self.seq_len));
            }
        }

        // Create tensors
        let inputs = Tensor::new(&inputs_data, &[self.batch_size, self.seq_len]);
        let targets = Tensor::new(&targets_data, &[self.batch_size, self.seq_len]);
        let mask = Tensor::new(&mask_data, &[self.batch_size, self.seq_len]);

        Ok(Some((inputs, targets, mask)))
    }

    /// Reset the dataloader to the beginning
    pub fn reset(&mut self) {
        self.current_pos = 0;
        self.shuffle();
    }

    /// Get the current dataloader state for checkpointing
    pub fn get_state(&self) -> DataLoaderState {
        DataLoaderState {
            current_pos: self.current_pos,
            rng_seed: self.rng_seed,
        }
    }

    /// Restore dataloader state from checkpoint
    pub fn restore_state(&mut self, state: &DataLoaderState) {
        self.current_pos = state.current_pos;
        // Note: RNG state restoration is approximate - we can't fully restore RNG state
        // This is acceptable per FR-020.2 (approximate dataloader state resumption)
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get number of instructions loaded
    pub fn instruction_count(&self) -> usize {
        self.conversations.len()
    }

    /// Shuffle conversations
    fn shuffle(&mut self) {
        let mut indices: Vec<usize> = (0..self.conversations.len()).collect();
        indices.shuffle(&mut self.rng);

        // Reorder both conversations and tokenized data
        let mut shuffled_conversations = Vec::new();
        let mut shuffled_tokenized = Vec::new();
        for &idx in &indices {
            shuffled_conversations.push(self.conversations[idx].clone());
            shuffled_tokenized.push(self.tokenized_conversations[idx].clone());
        }
        self.conversations = shuffled_conversations;
        self.tokenized_conversations = shuffled_tokenized;
    }
}
