# Quickstart Guide

**Date**: 2025-01-27  
**Phase**: 1 - Design & Contracts  
**Status**: Complete

## Overview

This guide provides step-by-step instructions for setting up and using the nanochat Rust port implementation.

## Prerequisites

- Rust 1.91.1 or later
- Cargo (comes with Rust)
- GPU with CUDA/Metal support (optional, for GPU acceleration)
- Training data in accessible format

## Workspace Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd nanochat
   ```

2. **Checkout the feature branch**:
   ```bash
   git checkout 001-rust-nanochat-port
   ```

3. **Build the workspace**:
   ```bash
   cargo build --release
   ```

   For GPU support:
   ```bash
   cargo build --release --features gpu
   ```

4. **Run tests**:
   ```bash
   cargo test --all
   ```

## Training Workflow

### Step 1: Prepare Training Data

Organize your training data into shards:

```bash
data/
├── shard_000.txt
├── shard_001.txt
├── shard_002.txt
└── ...
```

Each shard should contain plain text, one document per line or separated by special markers.

### Step 2: Train Tokenizer (Optional)

If you want to train a custom tokenizer:

```bash
cargo run --release --bin nanochat-tokenizer -- train \
  --data-dir ./data \
  --output ./tokenizer.apr \
  --vocab-size 50304
```

### Step 3: Pretraining

Create a training configuration file `pretrain.toml`:

```toml
[model]
vocab_size = 50304
n_layer = 20
n_head = 6
n_kv_head = 6
n_embd = 768
sequence_len = 2048
block_size = 2048
dropout = 0.1

[optimizer]
type = "AdamW"
learning_rate = 0.0003
weight_decay = 0.1

[training]
batch_size = 32
max_steps = 10000
checkpoint_interval = 1000
warmup_steps = 100
```

Run pretraining:

```bash
cargo run --release --bin nanochat-pretrain -- \
  --config pretrain.toml \
  --data-dir ./data \
  --output-dir ./output/pretrain
```

Monitor progress with indicatif progress bars. Checkpoints are saved to `./output/pretrain/checkpoints/`.

### Step 4: Mid-Training

Create mid-training configuration `midtrain.toml` (similar to pretrain.toml but with conversational data):

```bash
cargo run --release --bin nanochat-midtrain -- \
  --config midtrain.toml \
  --base-model ./output/pretrain/checkpoints/final.apr \
  --data-dir ./conversational_data \
  --output-dir ./output/midtrain
```

### Step 5: Supervised Fine-Tuning

Create SFT configuration `sft.toml`:

```bash
cargo run --release --bin nanochat-sft -- \
  --config sft.toml \
  --base-model ./output/midtrain/checkpoints/final.apr \
  --data-dir ./instruction_data \
  --output-dir ./output/sft
```

## Evaluation Workflow

Evaluate your trained model on standard benchmarks:

```bash
cargo run --release --bin nanochat-eval -- \
  --model ./output/sft/checkpoints/final.apr \
  --benchmarks all \
  --output-dir ./evaluation_results
```

This generates:
- `evaluation_results/report.md` - Comprehensive evaluation report
- `evaluation_results/benchmarks/*.json` - Individual benchmark results

## Inference Usage

### Command Line Interface

Generate text from a prompt:

```bash
cargo run --release --bin nanochat-cli -- infer \
  --model ./output/sft/checkpoints/final.apr \
  "Write a haiku about Rust programming"
```

With streaming (default):

```bash
cargo run --release --bin nanochat-cli -- infer \
  --model ./output/sft/checkpoints/final.apr \
  --stream \
  "Explain machine learning"
```

The output streams token-by-token with progress indicators.

### Web Interface

Start the inference server:

```bash
cargo run --release --bin nanochat-inference -- \
  --model ./output/sft/checkpoints/final.apr \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

Access the web interface at `http://localhost:8000` in your browser.

### REST API (OpenAI Compatible)

Send a chat completion request using OpenAI-compatible format:

```bash
curl -N -H "Content-Type: application/json" \
  -X POST \
  --data '{
    "model": "nanochat-d20",
    "messages": [
      {"role": "user", "content": "Write a haiku about Rust."}
    ],
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 256,
    "stream": true
  }' \
  http://localhost:8000/v1/chat/completions
```

The response streams as Server-Sent Events (SSE) in OpenAI-compatible format.

**Non-streaming request**:

```bash
curl -H "Content-Type: application/json" \
  -X POST \
  --data '{
    "model": "nanochat-d20",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Rust?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150,
    "stream": false
  }' \
  http://localhost:8000/v1/chat/completions
```

**List available models**:

```bash
curl http://localhost:8000/v1/models
```

The API is fully compatible with OpenAI's Chat Completions API, enabling drop-in replacement in OpenAI-compatible applications.

## Example: Complete Training Pipeline

Here's a complete example training a small model:

```bash
# 1. Pretraining
cargo run --release --bin nanochat-pretrain -- \
  --config configs/pretrain_d20.toml \
  --data-dir ./data/pretrain \
  --output-dir ./models/d20_pretrain

# 2. Mid-training
cargo run --release --bin nanochat-midtrain -- \
  --config configs/midtrain_d20.toml \
  --base-model ./models/d20_pretrain/checkpoints/final.apr \
  --data-dir ./data/conversational \
  --output-dir ./models/d20_midtrain

# 3. Supervised Fine-Tuning
cargo run --release --bin nanochat-sft -- \
  --config configs/sft_d20.toml \
  --base-model ./models/d20_midtrain/checkpoints/final.apr \
  --data-dir ./data/instructions \
  --output-dir ./models/d20_sft

# 4. Evaluation
cargo run --release --bin nanochat-eval -- \
  --model ./models/d20_sft/checkpoints/final.apr \
  --benchmarks all \
  --output-dir ./evaluation/d20

# 5. Inference
cargo run --release --bin nanochat-cli -- infer \
  --model ./models/d20_sft/checkpoints/final.apr \
  "Hello, how are you?"
```

## Troubleshooting

### GPU Not Detected

If GPU is not detected, check:
- GPU drivers are installed
- `gpu` feature is enabled: `cargo build --release --features gpu`
- Device selection: Use `--device gpu` explicitly

### Out of Memory

If you encounter OOM errors:
- Reduce `device_batch_size` in configuration
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use CPU backend: `--device cpu`

### Slow Training

To improve training speed:
- Enable GPU: `--features gpu --device gpu`
- Increase `workers` for data loading
- Use larger batch sizes (if memory allows)
- Enable SIMD optimizations (automatic with trueno)

### Checkpoint Loading Errors

If checkpoint loading fails:
- Verify checkpoint file exists and is readable
- Check checkpoint version compatibility
- Validate checksum: checkpoints include integrity checks

## Next Steps

- Read the full [specification](./spec.md) for detailed requirements
- Review the [implementation plan](./plan.md) for architecture details
- Check [research findings](./research.md) for technical decisions
- See [data model](./data-model.md) for entity definitions
- Review [API contracts](./contracts/) for interface specifications

