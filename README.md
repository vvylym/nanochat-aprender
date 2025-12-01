# Nanochat - Rust Port

A production-grade, 100% tested Rust port of [nanochat](https://github.com/karpathy/nanochat) (Karpathy's ChatGPT clone) using pure Rust machine learning frameworks.

## Overview

This project implements a complete language model training and inference pipeline in Rust, featuring:

- **Pure Rust**: Zero C/C++ dependencies - all ML operations use `aprender`, `trueno`, and `nalgebra`
- **Production Ready**: 100% test coverage, comprehensive error handling, and performance optimizations
- **Modular Architecture**: Multi-crate workspace with separate crates for each ML lifecycle stage
- **OpenAI Compatible**: Inference API fully compatible with OpenAI's Chat Completions API
- **GPU & CPU Support**: Accelerated training and inference via `wgpu` (GPU) and SIMD (CPU)

## Architecture

The project is organized as a Cargo workspace with the following crates:

- **`nanochat-model`**: Core GPT model implementation (attention, MLP, RoPE, RMSNorm)
- **`nanochat-tokenizer`**: BPE tokenizer for text preprocessing
- **`nanochat-pretrain`**: Pretraining stage for base language modeling
- **`nanochat-midtrain`**: Mid-training stage for conversational fine-tuning
- **`nanochat-sft`**: Supervised fine-tuning for instruction following
- **`nanochat-eval`**: Evaluation and benchmarking (CORE, ARC, GSM8K, HumanEval, MMLU, ChatCORE)
- **`nanochat-inference`**: OpenAI-compatible inference server (actix-web)
- **`nanochat-cli`**: Command-line interface with real-time progress (clap + indicatif)

## Quick Start

### Prerequisites

- Rust 1.91.1 or later
- Cargo (comes with Rust)
- GPU with CUDA/Metal support (optional, for GPU acceleration)

### Building

```bash
# Clone the repository
git clone <repository-url>
cd nanochat

# Build all crates
cargo build --release

# Build with GPU support
cargo build --release --features gpu
```

### Running Tests

```bash
# Run all tests
cargo test --all

# Run tests for a specific crate
cargo test -p nanochat-model
```

### Training

```bash
# Pretraining
cargo run --release --bin nanochat-pretrain -- \
    --data-dir ./data/pretrain \
    --checkpoint-dir ./checkpoints \
    --config configs/d20.toml

# Mid-training
cargo run --release --bin nanochat-midtrain -- \
    --data-dir ./data/midtrain \
    --checkpoint ./checkpoints/pretrain-final.apr \
    --output-dir ./checkpoints

# Supervised Fine-Tuning
cargo run --release --bin nanochat-sft -- \
    --data-dir ./data/sft \
    --checkpoint ./checkpoints/midtrain-final.apr \
    --output-dir ./checkpoints
```

### Inference

#### Command Line

```bash
# Generate text
cargo run --release --bin nanochat -- infer \
    --checkpoint ./checkpoints/sft-final.apr \
    --prompt "Hello, how are you?"

# Interactive chat
cargo run --release --bin nanochat -- chat \
    --checkpoint ./checkpoints/sft-final.apr
```

#### Inference Server

```bash
# Start the OpenAI-compatible inference server
cargo run --release --bin nanochat-inference -- \
    --checkpoint ./checkpoints/sft-final.apr \
    --host 0.0.0.0 \
    --port 8000

# Test with curl (OpenAI-compatible API)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nanochat",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    }'
```

### Evaluation

```bash
# Run all benchmarks
cargo run --release --bin nanochat-eval -- \
    --checkpoint ./checkpoints/sft-final.apr \
    --output ./eval-results.json

# Run specific benchmark
cargo run --release --bin nanochat-eval -- \
    --checkpoint ./checkpoints/sft-final.apr \
    --benchmark gsm8k \
    --output ./gsm8k-results.json
```

## Development

### Code Quality

The project enforces strict code quality standards:

- **Formatting**: `cargo fmt` (configured via `rustfmt.toml`)
- **Linting**: `cargo clippy` (configured via `.clippy.toml`)
- **Testing**: 100% test coverage requirement
- **Documentation**: All public APIs must be documented

### Running CI Checks Locally

```bash
# Format code
cargo fmt --all --check

# Run clippy
cargo clippy --all -- -D warnings

# Run tests
cargo test --all

# Check documentation
cargo doc --all --no-deps
```

## Performance Goals

- **Inference**: <5s for 100 tokens, <200ms first token latency
- **Training**: ≥1000 samples/sec per GPU device
- **Web**: ≥10 concurrent users without degradation
- **Streaming**: ≥10 tokens/sec throughput

## License

MIT

## Acknowledgments

- Original nanochat implementation by [Andrej Karpathy](https://github.com/karpathy/nanochat)
- Rust ML frameworks: `aprender`, `trueno`, `nalgebra`
- Web framework: `actix-web`
- CLI: `clap`, `indicatif`

