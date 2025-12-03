# Implementation Plan: Production-Grade Rust Port of Nanochat

**Branch**: `001-rust-nanochat-port` | **Date**: 2025-12-01 | **Last Updated**: 2025-12-03 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-rust-nanochat-port/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This plan implements a production-grade, 100% tested Rust port of nanochat (Karpathy's ChatGPT clone) using the aprender ML framework with trueno and nalgebra. The implementation follows LLMOps best practices, organizing development into distinct phases that map to the complete machine learning lifecycle: data preparation, model development, training pipeline, evaluation, deployment, and serving. The system is organized as a modular Cargo workspace with separate crates for each ML cycle step, ensuring maintainability, testability, and production readiness. **The inference API is fully OpenAI-compatible**, enabling seamless integration with existing OpenAI-compatible tools and libraries, making it a drop-in replacement for OpenAI's Chat Completions API.

## Technical Context

**Language/Version**: Rust 1.91.1 (edition 2021)  
**Primary Dependencies**: 
- ML: aprender (fork: @aprender), trueno (fork: @trueno), nalgebra (fork: @nalgebra) - all pure Rust
- Web: actix-web, tokio, serde
- CLI: clap, indicatif
- All dependencies are pure Rust with zero C/C++ FFI

**Storage**: File-based checkpoint storage (SafeTensors format via aprender for model weights, JSON for metadata), tokenizer storage (JSON format `tokenizer.json`), in-memory caching for inference  
**Testing**: cargo test with 100% coverage requirement, criterion for benchmarks, proptest for property-based testing  
**Target Platform**: Linux/macOS/Windows (native), WebAssembly (future via aprender WASM support)  
**Project Type**: Multi-crate workspace (8 crates: model, tokenizer, pretrain, midtrain, sft, eval, inference, cli)  
**Performance Goals**: 
- Inference: <5s for 100 tokens, <200ms first token latency
- Training: ≥1000 samples/sec per GPU device
- Web: ≥10 concurrent users without degradation
- Streaming: ≥10 tokens/sec throughput

**Constraints**: 
- Zero C/C++ dependencies (pure Rust only)
- 100% test coverage for public APIs
- Numerical parity with reference implementation (within tolerance)
- Memory safety: no unsafe code without documented safety invariants
- Compile without warnings

**Scale/Scope**: 
- Model sizes: d20 (speedrun) to d32 (full training)
- Vocabulary: 50K+ tokens
- Context window: up to 2048 tokens
- Training data: multi-billion token datasets
- Concurrent inference: 10+ simultaneous users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Pure Rust (Zero C/C++ Dependencies)**
- [x] All dependencies are pure Rust crates (aprender, trueno, nalgebra, actix-web, tokio, serde, clap, indicatif - no FFI, no C/C++ compilation)
- [x] No build-time C/C++ steps required
- [x] All ML operations use Rust-native implementations (aprender with trueno/nalgebra backends)

**II. Performance & Safety First**
- [ ] Unsafe code usage documented with safety invariants (if any) - will be verified during implementation
- [ ] Performance-critical paths identified and benchmarked - will be identified in Phase 0 research
- [ ] Memory safety verified (ownership, no data races, no buffer overflows) - Rust type system enforces this

**III. Minimalism & Simplicity**
- [x] No excessive abstraction layers or configuration objects - modular crate structure with single-purpose crates
- [x] Each module has clear, single purpose - each crate handles one ML cycle step
- [x] Code is explicit and readable (no clever abstractions) - will be enforced via code review
- [x] YAGNI principle applied (no premature features) - only nanochat reference features implemented

**IV. Classical ML Algorithms**
- [x] Algorithm choices documented with rationale - documented in research.md with references
- [x] Numerical stability considerations addressed - documented in research.md: max-subtraction, QK-norm, gradient clipping
- [x] Performance optimizations for Rust characteristics - trueno SIMD, aprender GPU support, identified in research.md

**V. Code Quality & Readability**
- [x] Public APIs have comprehensive documentation - requirement FR-071, will be enforced during implementation
- [x] Naming follows Rust idioms - requirement FR-072, will be enforced via code review
- [x] Code formatting enforced (rustfmt) - requirement FR-073, automated in CI
- [x] Complex logic includes inline comments - requirement FR-075, will be enforced during implementation

**VI. LLM Disclosure Policy**
- [ ] Any LLM-assisted code sections declared in PR - will be enforced via PR template

## Project Structure

### Documentation (this feature)

```text
specs/001-rust-nanochat-port/
├── plan.md                   # This file (/speckit.plan command output)
├── research.md               # Phase 0 output (/speckit.plan command)
├── data-model.md             # Phase 1 output (/speckit.plan command)
├── quickstart.md             # Phase 1 output (/speckit.plan command)
├── contracts/                # Phase 1 output (/speckit.plan command)
│   ├── inference-api.yaml    # REST API contract for inference server
│   └── training-api.md       # CLI contract for training commands
└── tasks.md                  # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
workspace-root/
├── Cargo.toml                    # Workspace configuration
├── README.md
├── crates/
│   ├── nanochat-model/           # Core GPT model implementation
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs            # Public API
│   │   │   ├── config.rs         # Model configuration
│   │   │   ├── gpt.rs            # GPT architecture
│   │   │   ├── attention.rs      # Multi-head attention with GQA
│   │   │   ├── mlp.rs            # MLP with ReLU²
│   │   │   ├── rope.rs           # Rotary position embeddings
│   │   │   ├── norm.rs           # RMSNorm implementation
│   │   │   ├── checkpoint.rs     # Checkpoint save/load (SafeTensors format)
│   │   │   ├── init.rs           # Weight initialization helpers
│   │   │   └── stability.rs      # Numerical stability checks
│   │   └── tests/
│   │       ├── unit/
│   │       └── integration/
│   │
│   ├── nanochat-tokenizer/       # BPE tokenizer
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs            # Tokenizer API wrapping aprender's BpeTokenizer
│   │   └── tests/
│   │
│   ├── nanochat-pretrain/       # Pretraining stage
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs           # CLI entry point
│   │   │   ├── train.rs          # Training loop
│   │   │   ├── optimizer.rs      # AdamW optimizer
│   │   │   ├── dataloader.rs     # Data loading and batching
│   │   │   └── metrics.rs        # Training metrics
│   │   └── tests/
│   │
│   ├── nanochat-midtrain/       # Mid-training stage
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── train.rs
│   │   │   └── dataloader.rs
│   │   └── tests/
│   │
│   ├── nanochat-sft/            # Supervised fine-tuning
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── train.rs
│   │   │   └── dataloader.rs
│   │   └── tests/
│   │
│   ├── nanochat-eval/           # Evaluation and benchmarking
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── core.rs          # CORE benchmark
│   │   │   ├── arc.rs           # ARC benchmarks
│   │   │   ├── gsm8k.rs         # GSM8K benchmark
│   │   │   ├── humaneval.rs     # HumanEval benchmark
│   │   │   ├── mmlu.rs          # MMLU benchmark
│   │   │   ├── chatcore.rs      # ChatCORE benchmark
│   │   │   └── report.rs        # Report generation
│   │   └── tests/
│   │
│   ├── nanochat-inference/      # Inference server (actix-web)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── server.rs         # Actix-web server setup
│   │   │   ├── handlers.rs       # HTTP handlers
│   │   │   ├── cache.rs         # Inference caching
│   │   │   ├── streaming.rs     # SSE streaming
│   │   │   └── session.rs       # Conversation session management
│   │   └── tests/
│   │       └── contract/        # Contract tests for API
│   │
│   └── nanochat-cli/           # CLI interface
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs
│       │   ├── commands/        # CLI commands (train, eval, infer)
│       │   │   ├── train.rs
│       │   │   ├── eval.rs
│       │   │   └── infer.rs
│       │   └── ui.rs            # Indicatif progress bars
│       └── tests/
│
└── tests/                       # Workspace-level integration tests
    ├── integration/
    └── benchmarks/
```

**Structure Decision**: Multi-crate workspace following LLMOps best practices. Each ML cycle step (pretraining, mid-training, SFT, evaluation) is a separate crate, enabling independent development, testing, and deployment. The model crate provides the core architecture, tokenizer wraps aprender's BPE implementation (no custom BPE code - uses aprender API per Principle VII), inference server handles deployment, and CLI provides user interface. This structure aligns with the constitution's minimalism principle while maintaining clear separation of concerns.

**Implementation Status**:
- **Phase 2 (Tokenizer)**: Complete - Uses aprender's `BpeTokenizer` directly, serializes to JSON format
- **Phase 3 (Model)**: Complete - Core GPT architecture with SafeTensors checkpoint format
- **Phase 4-6 (Training)**: Crate structure exists, placeholder implementations ready for development
- **Phase 7 (Evaluation)**: Crate structure exists, placeholder implementations ready for development
- **Phase 8-9 (Inference)**: Crate structure exists, placeholder implementations ready for development

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations identified. All dependencies are pure Rust, architecture is minimal and modular, and the structure follows best practices without unnecessary complexity.

## LLMOps Workflow Phases

This implementation follows LLMOps best practices, organizing development into phases that map to the complete machine learning lifecycle:

### Phase 0: Research & Architecture (Foundation)
**LLMOps Step**: Architecture Design & Technology Selection
- Research aprender capabilities for transformer implementation
- Research trueno/nalgebra for tensor operations
- Design model architecture and data flow
- Establish testing and benchmarking strategies
- Document algorithm choices and numerical stability considerations

### Phase 1: Data Preparation & Preprocessing
**LLMOps Step**: Data Pipeline Development
- Integrate BPE tokenizer using aprender's `BpeTokenizer` (no custom implementation needed - uses aprender API per Principle VII)
- Implement tokenizer save/load using JSON format (`tokenizer.json`) - serializes vocabulary and merges
- Implement data loading and preprocessing
- Implement data sharding for distributed training
- Establish data validation and quality checks

**Architecture Note**: The tokenizer crate wraps aprender's `BpeTokenizer` directly, providing a clean API while leveraging aprender's battle-tested BPE implementation. Special tokens, vocabulary, and encoding/decoding are all handled by aprender's API.

### Phase 2: Model Architecture Development
**LLMOps Step**: Model Implementation
- Implement core GPT architecture (decoder-only transformer)
- Implement attention mechanisms (GQA, RoPE, QK-norm)
- Implement MLP layers (ReLU² activation)
- Implement normalization (RMSNorm)
- Implement checkpoint save/load using SafeTensors format (aprender's standard)

#### Device Management

**Architecture Decision**: The `get_device()` method from Python reference is not implemented. Instead:

- **Aprender Feature Flags**: Device selection is compile-time via `gpu` feature flag
- **Model Crate**: Works with aprender's tensor operations (CPU or GPU via feature flags)
- **No Runtime Device Selection**: Unlike PyTorch, aprender uses feature flags for device selection

**Python Reference Alignment**: Python's `GPT.get_device()` returns the device (CPU/GPU) of model parameters. In Rust:
- Device is determined at compile time via `gpu` feature flag
- All tensors use the same backend (CPU via trueno, or GPU via wgpu)
- No runtime device switching needed (simpler, more Rust-idiomatic)

### Phase 3: Training Pipeline Development
**LLMOps Step**: Training Infrastructure
- Implement pretraining stage (base language modeling)
- Implement mid-training stage (conversational fine-tuning)
- Implement supervised fine-tuning (SFT) stage
- Implement optimizers (AdamW with learning rate scheduling)
- Implement distributed training support
- Implement checkpoint management and resumption

#### Optimizer Setup

**Architecture Decision**: The `setup_optimizers()` method from Python reference is not implemented in the model crate. Instead:

- **Model Crate** (`nanochat-model`): Provides core architecture only
- **Training Crates** (`nanochat-pretrain`, `nanochat-midtrain`, `nanochat-sft`): Implement optimizer setup using aprender's optimizers

This separation follows the plan's architecture where:
- Model crate = Core architecture (no training logic)
- Training crates = Training loops, optimizers, schedulers

**Python Reference Alignment**: Python's `GPT.setup_optimizers()` configures separate optimizers for different parameter groups. In Rust:
- Model provides `parameters()` and `parameters_mut()` methods
- Training crates use aprender's `AdamW` and schedulers to configure optimizers
- Parameter grouping (embedding, LM head, matrix params) is handled in training crates

### Phase 4: Evaluation & Validation
**LLMOps Step**: Model Evaluation & Benchmarking
- Implement CORE benchmark evaluation
- Implement ARC (Easy/Challenge) benchmarks
- Implement GSM8K math reasoning benchmark
- Implement HumanEval code generation benchmark
- Implement MMLU knowledge benchmark
- Implement ChatCORE conversational benchmark
- Implement evaluation report generation

### Phase 5: Deployment & Serving
**LLMOps Step**: Model Deployment & Inference
- Implement inference server (actix-web with tokio)
- Implement OpenAI-compatible REST API endpoints (`/v1/chat/completions`, `/v1/models`)
- Implement Server-Sent Events (SSE) streaming with OpenAI-compatible format
- Implement non-streaming responses with OpenAI-compatible JSON structure
- Implement inference caching
- Implement conversation session management
- Implement CLI interface with indicatif progress bars
- Implement streaming text generation
- Ensure full OpenAI API compatibility for seamless integration

#### Model Generation Methods

**Architecture Decision**: The `generate()` method from Python reference is not implemented in the model crate. Instead:

- **Model Crate** (`nanochat-model`): Provides core forward pass with KV cache support via `forward_cache()` method
- **CLI Crate** (`nanochat-cli`): Implements `generate()` for command-line text generation
- **Inference Server** (`nanochat-inference`): Implements generation for web API

This separation follows the plan's architecture where:
- Model crate = Core architecture (forward pass, checkpoints)
- Training crates = Training loops and optimizer setup
- Inference crates = Text generation and sampling strategies

**Python Reference Alignment**: Python's `GPT.generate()` combines model forward pass with sampling. In Rust, this is split:
- Model provides `forward_cache()` with KV cache
- Inference crates implement sampling (greedy, temperature, top-k, top-p)

### Phase 6: Integration & Testing
**LLMOps Step**: End-to-End Validation
- Integration tests for complete training pipeline
- Integration tests for inference workflows
- Performance benchmarking
- Numerical parity validation with reference implementation
- Load testing for concurrent users
- Documentation and quickstart guides

## Phase 0: Research & Architecture

**Goal**: Establish technical foundation, research dependencies, and design architecture following LLMOps best practices.

### Research Tasks

1. **Aprender Framework Capabilities**
   - Research aprender's neural network and transformer support
   - Verify GPU support via `gpu` feature (wgpu backend)
   - Document available optimizers (AdamW support)
   - Identify gaps requiring trueno/nalgebra direct usage

2. **Trueno & Nalgebra Integration**
   - Research trueno SIMD capabilities for CPU acceleration
   - Research trueno GPU support via wgpu
   - Research nalgebra linear algebra operations needed
   - Document integration patterns with aprender

3. **Model Architecture Design**
   - Design GPT architecture using aprender primitives
   - Design attention mechanism with GQA support
   - Design RoPE implementation strategy
   - Design checkpoint format compatibility

4. **Numerical Stability & Performance**
   - Research numerical stability best practices for Rust ML
   - Identify performance-critical paths
   - Design benchmarking strategy
   - Document optimization opportunities

5. **Testing & Quality Assurance**
   - Design test strategy for 100% coverage requirement
   - Design property-based testing approach
   - Design numerical parity validation approach
   - Design integration testing strategy

**Output**: `research.md` with all technical decisions documented

## Phase 1: Design & Contracts

**Prerequisites**: Phase 0 research complete

### Data Model

Extract entities from feature spec:
- Model Checkpoint (weights, config, metadata)
- Tokenizer (vocabulary, special tokens, encoding/decoding)
- Training Configuration (hyperparameters, architecture params)
- Evaluation Result (benchmark scores, metrics)
- Conversation Session (message history, context window)
- Training Dataset (data sources, preprocessing)

**Output**: `data-model.md` with entity definitions, relationships, and validation rules

### API Contracts

Generate contracts from functional requirements:

1. **Inference Server API** (REST)
   - POST /chat/completions - Chat completion with streaming
   - GET /health - Health check
   - GET /stats - Server statistics
   - POST /cache/invalidate - Cache invalidation

2. **Training CLI Contracts**
   - `nanochat pretrain` - Pretraining command
   - `nanochat midtrain` - Mid-training command
   - `nanochat sft` - Supervised fine-tuning command
   - `nanochat eval` - Evaluation command
   - `nanochat infer` - Inference command

**Output**: `contracts/inference-api.yaml` (OpenAPI), `contracts/training-api.yaml` (CLI spec)

### Quickstart Guide

Create quickstart guide demonstrating:
- Workspace setup
- Model training workflow
- Inference usage (CLI and web)
- Evaluation workflow

**Output**: `quickstart.md` with step-by-step examples

### Agent Context Update

Run `.specify/scripts/bash/update-agent-context.sh cursor-agent` to update agent-specific context with new technologies (aprender, trueno, nalgebra, actix-web).

**Output**: Updated agent context file

## Next Steps

After Phase 1 completion:
1. Run `/speckit.tasks` to generate detailed task breakdown
2. Begin Phase 2 implementation following LLMOps workflow
3. Each phase should be independently testable and deliverable
