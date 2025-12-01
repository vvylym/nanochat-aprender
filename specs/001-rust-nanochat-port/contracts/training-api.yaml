# CLI API Contract for Nanochat Training Commands
# Version: 1.0.0
# Date: 2025-01-27
#
# Overview:
#   This document defines the CLI interface contracts for nanochat training,
#   evaluation, and inference commands.
#
# Commands:

### nanochat pretrain

Pretraining stage for base language modeling.

**Usage**: `nanochat pretrain [OPTIONS]`

**Options**:
- `--config <PATH>` - Path to training configuration file (required)
- `--data-dir <PATH>` - Directory containing training data shards (required)
- `--output-dir <PATH>` - Directory for checkpoints and outputs (required)
- `--resume <PATH>` - Path to checkpoint to resume from (optional)
- `--device <DEVICE>` - Device to use: `cpu`, `gpu`, or `auto` (default: `auto`)
- `--workers <N>` - Number of data loading workers (default: 4)
- `--log-interval <N>` - Steps between logging metrics (default: 100)
- `--checkpoint-interval <N>` - Steps between checkpoint saves (default: 1000)
- `--wandb-project <NAME>` - Weights & Biases project name (optional)
- `--quiet` - Suppress progress output (default: false)

**Exit Codes**:
- `0` - Success
- `1` - Configuration error
- `2` - Data loading error
- `3` - Training error
- `4` - Checkpoint error

**Output**:
- Checkpoints saved to `{output-dir}/checkpoints/step_{N}.apr`
- Training metrics logged to console (or wandb if configured)
- Final checkpoint at `{output-dir}/checkpoints/final.apr`

### nanochat midtrain

Mid-training stage for conversational fine-tuning.

**Usage**: `nanochat midtrain [OPTIONS]`

**Options**:
- `--config <PATH>` - Path to training configuration file (required)
- `--base-model <PATH>` - Path to pretrained base model checkpoint (required)
- `--data-dir <PATH>` - Directory containing conversational training data (required)
- `--output-dir <PATH>` - Directory for checkpoints and outputs (required)
- `--resume <PATH>` - Path to checkpoint to resume from (optional)
- `--device <DEVICE>` - Device to use: `cpu`, `gpu`, or `auto` (default: `auto`)
- `--workers <N>` - Number of data loading workers (default: 4)
- `--log-interval <N>` - Steps between logging metrics (default: 100)
- `--checkpoint-interval <N>` - Steps between checkpoint saves (default: 1000)
- `--wandb-project <NAME>` - Weights & Biases project name (optional)
- `--quiet` - Suppress progress output (default: false)

**Exit Codes**: Same as `pretrain`

**Output**: Same format as `pretrain`

### nanochat sft

Supervised fine-tuning stage for instruction following.

**Usage**: `nanochat sft [OPTIONS]`

**Options**:
- `--config <PATH>` - Path to training configuration file (required)
- `--base-model <PATH>` - Path to mid-trained model checkpoint (required)
- `--data-dir <PATH>` - Directory containing instruction-following data (required)
- `--output-dir <PATH>` - Directory for checkpoints and outputs (required)
- `--resume <PATH>` - Path to checkpoint to resume from (optional)
- `--device <DEVICE>` - Device to use: `cpu`, `gpu`, or `auto` (default: `auto`)
- `--workers <N>` - Number of data loading workers (default: 4)
- `--log-interval <N>` - Steps between logging metrics (default: 100)
- `--checkpoint-interval <N>` - Steps between checkpoint saves (default: 1000)
- `--wandb-project <NAME>` - Weights & Biases project name (optional)
- `--quiet` - Suppress progress output (default: false)

**Exit Codes**: Same as `pretrain`

**Output**: Same format as `pretrain`

### nanochat eval

Evaluate model on standard benchmarks.

**Usage**: `nanochat eval [OPTIONS]`

**Options**:
- `--model <PATH>` - Path to model checkpoint to evaluate (required)
- `--benchmarks <LIST>` - Comma-separated list of benchmarks: `core,arc,gsm8k,humaneval,mmlu,chatcore,all` (default: `all`)
- `--output-dir <PATH>` - Directory for evaluation results (required)
- `--device <DEVICE>` - Device to use: `cpu`, `gpu`, or `auto` (default: `auto`)
- `--batch-size <N>` - Batch size for evaluation (default: 32)
- `--quiet` - Suppress progress output (default: false)

**Exit Codes**:
- `0` - Success
- `1` - Model loading error
- `2` - Benchmark execution error
- `3` - Report generation error

**Output**:
- Evaluation report at `{output-dir}/report.md`
- Individual benchmark results in `{output-dir}/benchmarks/{name}.json`

### nanochat infer

Run inference from command line.

**Usage**: `nanochat infer [OPTIONS] [PROMPT]`

**Arguments**:
- `PROMPT` - Text prompt for generation (required if not provided via stdin)

**Options**:
- `--model <PATH>` - Path to model checkpoint (required)
- `--temperature <FLOAT>` - Sampling temperature (default: 0.8, range: 0.0-2.0)
- `--top-k <INT>` - Top-k sampling (default: 50, range: 1-200)
- `--top-p <FLOAT>` - Nucleus sampling (default: 0.9, range: 0.0-1.0)
- `--max-tokens <INT>` - Maximum tokens to generate (default: 256, range: 1-4096)
- `--seed <INT>` - Random seed for reproducibility (optional)
- `--device <DEVICE>` - Device to use: `cpu`, `gpu`, or `auto` (default: `auto`)
- `--stream` - Enable streaming output (default: true)
- `--no-progress` - Disable progress bars (default: false)

**Exit Codes**:
- `0` - Success
- `1` - Model loading error
- `2` - Inference error

**Output**:
- Generated text to stdout
- Progress bars via indicatif (if enabled)
- Streaming tokens as they're generated (if enabled)

## Configuration File Format

Training configuration files use TOML format:

```toml
[model]
vocab_size = 50304
n_layer = 20
n_head = 6
n_kv_head = 6
n_embd = 768
sequence_len = 2048
block_size = 2048
dropout = 0.0

[optimizer]
type = "AdamW"
learning_rate = 0.0003
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8

[training]
batch_size = 32
device_batch_size = 32
gradient_accumulation_steps = 1
max_steps = 10000
checkpoint_interval = 1000
warmup_steps = 100

[schedule]
type = "WarmupCosine"
warmup_steps = 100
total_steps = 10000
```

## Progress Output Format

All training commands use indicatif for progress display:

```
Training: [████████████████████] 45% (4500/10000 steps)
Loss: 2.345 | LR: 0.0003 | Throughput: 1250 samples/s
```

Inference uses streaming output with progress:

```
Generating: [████████░░░░░░░░░░] 50% (128/256 tokens)
```

## Error Messages

All commands provide clear error messages:

- Configuration errors: "Invalid configuration: {reason}"
- Model loading errors: "Failed to load model from {path}: {error}"
- Training errors: "Training failed at step {N}: {error}"
- Data errors: "Failed to load data from {path}: {error}"

