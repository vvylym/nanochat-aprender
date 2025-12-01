# Data Model

**Date**: 2025-01-27  
**Phase**: 1 - Design & Contracts  
**Status**: Complete

## Overview

This document defines the core entities, their attributes, relationships, and validation rules for the nanochat Rust port implementation.

## Entities

### ModelCheckpoint

Represents a saved state of a trained model, containing all information needed to resume training or perform inference.

**Attributes**:
- `weights: HashMap<String, Tensor>` - Model parameter tensors keyed by parameter name
- `config: ModelConfig` - Model architecture configuration (depth, width, attention heads, etc.)
- `metadata: CheckpointMetadata` - Training metadata (step count, loss values, learning rate, etc.)
- `version: String` - Checkpoint format version for compatibility checking
- `checksum: u32` - CRC32 checksum for integrity validation

**Relationships**:
- Used by: `TrainingSession`, `InferenceSession`
- Produced by: `PretrainingStage`, `MidTrainingStage`, `SFTStage`

**Validation Rules**:
- All required weights must be present
- Config must match model architecture
- Checksum must validate on load
- Version must be compatible with current implementation

**State Transitions**:
- `Created` → `Saved` (checkpoint written to disk)
- `Saved` → `Loaded` (checkpoint loaded into memory)
- `Loaded` → `Validated` (integrity checks passed)

### ModelConfig

Represents model architecture hyperparameters and configuration.

**Attributes**:
- `vocab_size: usize` - Tokenizer vocabulary size
- `n_layer: usize` - Number of transformer layers (depth)
- `n_head: usize` - Number of query attention heads
- `n_kv_head: usize` - Number of key-value attention heads (GQA)
- `n_embd: usize` - Embedding dimension (model width)
- `sequence_len: usize` - Maximum sequence length (context window)
- `block_size: usize` - Context window size (typically equals sequence_len)
- `dropout: f32` - Dropout probability (0.0 for inference)

**Relationships**:
- Part of: `ModelCheckpoint`
- Used by: `GPTModel`, all training stages

**Validation Rules**:
- `n_embd % n_head == 0` (embedding must be divisible by heads)
- `n_head % n_kv_head == 0` (query heads must be divisible by KV heads for GQA)
- `n_kv_head <= n_head` (KV heads cannot exceed query heads)
- `sequence_len > 0` and `block_size > 0`
- `dropout >= 0.0 && dropout <= 1.0`

### Tokenizer

Represents the text tokenization component that converts between text and token sequences.

**Attributes**:
- `vocab: HashMap<String, u32>` - Token to ID mapping
- `vocab_inverse: HashMap<u32, String>` - ID to token mapping
- `special_tokens: SpecialTokens` - Special token definitions (BOS, EOS, PAD, etc.)
- `vocab_size: usize` - Total vocabulary size
- `merges: Vec<(String, String)>` - BPE merge rules (for BPE tokenizers)

**Relationships**:
- Used by: All training stages, inference, data loading
- Produced by: `TokenizerTraining`

**Validation Rules**:
- Vocab and vocab_inverse must be consistent (bidirectional mapping)
- All special tokens must have valid IDs in vocab
- Vocab size must match vocab map size
- BPE merges must be valid (if applicable)

**State Transitions**:
- `Untrained` → `Training` → `Trained` → `Loaded`

### TrainingConfiguration

Represents hyperparameters and settings for model training.

**Attributes**:
- `model_config: ModelConfig` - Model architecture configuration
- `optimizer_config: OptimizerConfig` - Optimizer settings (learning rate, weight decay, etc.)
- `batch_size: usize` - Training batch size
- `device_batch_size: usize` - Batch size per device (for distributed training)
- `gradient_accumulation_steps: usize` - Steps for gradient accumulation
- `max_steps: usize` - Maximum training steps
- `checkpoint_interval: usize` - Steps between checkpoint saves
- `learning_rate_schedule: LRSchedule` - Learning rate scheduling configuration
- `warmup_steps: usize` - Learning rate warmup steps

**Relationships**:
- Used by: All training stages
- Contains: `ModelConfig`, `OptimizerConfig`

**Validation Rules**:
- Batch sizes must be positive
- Gradient accumulation steps must be positive
- Learning rate must be positive
- Warmup steps must be <= max_steps

### OptimizerConfig

Represents optimizer hyperparameters.

**Attributes**:
- `optimizer_type: OptimizerType` - AdamW, SGD, etc.
- `learning_rate: f32` - Base learning rate
- `weight_decay: f32` - Weight decay coefficient
- `beta1: f32` - Adam beta1 parameter
- `beta2: f32` - Adam beta2 parameter
- `eps: f32` - Epsilon for numerical stability

**Relationships**:
- Part of: `TrainingConfiguration`

**Validation Rules**:
- Learning rate > 0
- Weight decay >= 0
- Beta values in (0, 1) range
- Epsilon > 0

### EvaluationResult

Represents benchmark scores and metrics from model evaluation runs.

**Attributes**:
- `model_checkpoint: PathBuf` - Path to evaluated checkpoint
- `benchmark_scores: HashMap<String, f32>` - Benchmark name to score mapping
- `evaluation_date: DateTime<Utc>` - When evaluation was performed
- `metrics: HashMap<String, f32>` - Additional metrics (loss, perplexity, etc.)
- `config: ModelConfig` - Model configuration used

**Relationships**:
- Produced by: `EvaluationStage`
- References: `ModelCheckpoint`

**Validation Rules**:
- All benchmark scores must be in valid range (typically [0, 1] for accuracy)
- Evaluation date must be valid
- Model checkpoint must exist

### ConversationSession

Represents an active chat interaction with message history and context.

**Attributes**:
- `session_id: String` - Unique session identifier
- `messages: Vec<Message>` - Conversation message history
- `context_window: Vec<u32>` - Token IDs in current context
- `generation_params: GenerationParams` - Current generation parameters
- `created_at: DateTime<Utc>` - Session creation time
- `last_activity: DateTime<Utc>` - Last activity timestamp

**Relationships**:
- Used by: `InferenceServer`
- Contains: `Message`, `GenerationParams`

**Validation Rules**:
- Context window must not exceed model's max sequence length
- Messages must be in chronological order
- Session ID must be unique

### Message

Represents a single message in a conversation.

**Attributes**:
- `role: MessageRole` - user, assistant, or system
- `content: String` - Message text content
- `timestamp: DateTime<Utc>` - Message timestamp
- `token_count: usize` - Number of tokens in message

**Relationships**:
- Part of: `ConversationSession`

**Validation Rules**:
- Content must not be empty
- Role must be valid (user, assistant, system)
- Token count must match content tokenization

### GenerationParams

Represents parameters for text generation.

**Attributes**:
- `temperature: f32` - Sampling temperature
- `top_k: usize` - Top-k sampling parameter
- `top_p: f32` - Nucleus sampling parameter
- `max_tokens: usize` - Maximum tokens to generate
- `seed: Option<u64>` - Random seed for reproducibility

**Relationships**:
- Part of: `ConversationSession`
- Used by: `InferenceEngine`

**Validation Rules**:
- Temperature >= 0.0 && <= 2.0
- Top-k >= 1 && <= vocab_size
- Top-p > 0.0 && <= 1.0
- Max tokens > 0 && <= context window remaining

### TrainingDataset

Represents the corpus of text used for training.

**Attributes**:
- `sources: Vec<DataSource>` - Data source definitions
- `preprocessing_steps: Vec<PreprocessingStep>` - Applied preprocessing
- `total_tokens: usize` - Total token count
- `shards: Vec<PathBuf>` - Paths to data shard files
- `metadata: DatasetMetadata` - Dataset metadata (name, version, etc.)

**Relationships**:
- Used by: All training stages
- Contains: `DataSource`, `PreprocessingStep`

**Validation Rules**:
- At least one data source must be present
- All shard files must exist
- Total tokens must match sum of shard tokens

### DataSource

Represents a single data source for training.

**Attributes**:
- `source_type: SourceType` - File, URL, HuggingFace dataset, etc.
- `path: String` - Path or identifier
- `format: DataFormat` - Text, JSONL, etc.
- `weight: f32` - Sampling weight (for multi-source training)

**Relationships**:
- Part of: `TrainingDataset`

**Validation Rules**:
- Path must be valid
- Weight must be positive
- Format must be supported

## Relationships Summary

```
ModelCheckpoint
  ├── contains ModelConfig
  ├── used by TrainingSession
  └── used by InferenceSession

TrainingConfiguration
  ├── contains ModelConfig
  └── contains OptimizerConfig

ConversationSession
  ├── contains Message[]
  └── contains GenerationParams

TrainingDataset
  ├── contains DataSource[]
  └── contains PreprocessingStep[]

EvaluationResult
  └── references ModelCheckpoint
```

## Validation Rules Summary

1. **Model Architecture**: All dimension constraints must be satisfied (divisibility, ranges)
2. **Checkpoint Integrity**: Checksums must validate, versions must be compatible
3. **Context Limits**: Context windows must not exceed model's max sequence length
4. **Parameter Ranges**: All hyperparameters must be in valid ranges
5. **Data Consistency**: Vocab mappings, message ordering, token counts must be consistent
6. **File Existence**: Referenced files (checkpoints, datasets) must exist

## State Machines

### Checkpoint Lifecycle
```
Created → Saved → Loaded → Validated → (In Use) → Saved (updated)
```

### Training Session Lifecycle
```
Initialized → Training → Checkpoint Saved → Training → ... → Completed
                                                      ↓
                                                  Interrupted
                                                      ↓
                                                  Resumed
```

### Inference Session Lifecycle
```
Created → Model Loaded → Ready → Processing → Streaming → Complete
                                                      ↓
                                                  Error
```

