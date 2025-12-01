# Tasks: Production-Grade Rust Port of Nanochat

**Input**: Design documents from `/specs/001-rust-nanochat-port/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Organization**: Tasks are organized by LLMOps workflow phases, with each phase corresponding to a specific LLMOps step and focusing on a single crate (or related crates). Phases are sequential - each depends only on previous phases.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Workspace root**: Repository root with `Cargo.toml` workspace configuration
- **Crates**: `crates/{crate-name}/src/` for each crate
- **Tests**: `crates/{crate-name}/tests/` for crate-specific tests, `tests/` for workspace-level integration tests

---

## Phase 1: Setup (Workspace Initialization)

**LLMOps Step**: Project Setup & Infrastructure  
**Purpose**: Initialize Cargo workspace and configure development environment  
**Crate**: Workspace-level setup

- [X] T001 Create workspace root `Cargo.toml` with workspace members configuration in `/home/gmi/capstone/nanochat/Cargo.toml`
- [X] T002 [P] Create `crates/nanochat-model/Cargo.toml` with aprender, trueno, nalgebra dependencies
- [X] T003 [P] Create `crates/nanochat-tokenizer/Cargo.toml` with aprender dependencies
- [X] T004 [P] Create `crates/nanochat-pretrain/Cargo.toml` with aprender, clap dependencies
- [X] T005 [P] Create `crates/nanochat-midtrain/Cargo.toml` with aprender, clap dependencies
- [X] T006 [P] Create `crates/nanochat-sft/Cargo.toml` with aprender, clap dependencies
- [X] T007 [P] Create `crates/nanochat-eval/Cargo.toml` with aprender, clap dependencies
- [X] T008 [P] Create `crates/nanochat-inference/Cargo.toml` with aprender, actix-web, tokio, serde dependencies
- [X] T009 [P] Create `crates/nanochat-cli/Cargo.toml` with aprender, clap, indicatif dependencies
- [X] T010 Configure rustfmt.toml at workspace root for consistent formatting
- [X] T011 Configure .clippy.toml at workspace root with appropriate lints
- [X] T012 Create workspace-level README.md with project overview and quickstart
- [X] T013 [P] Create `.github/workflows/ci.yml` for continuous integration (tests, linting, formatting)
- [X] T014 [P] Create `.github/workflows/benchmarks.yml` for performance benchmarking

**Checkpoint**: Workspace structure ready, all crates initialized, CI configured

---

## Phase 2: Data Preparation & Preprocessing (LLMOps: Data Pipeline Development)

**LLMOps Step**: Data Pipeline Development  
**Purpose**: Implement BPE tokenizer for text preprocessing - foundational for all ML operations  
**Crate**: `nanochat-tokenizer`  
**Dependencies**: Phase 1 (Setup)

### Tests for Tokenizer (Write First)

- [X] T015 [P] [US5] Create unit test for BPE training in `crates/nanochat-tokenizer/tests/test_bpe.rs`
- [X] T016 [P] [US5] Create unit test for encoding/decoding in `crates/nanochat-tokenizer/tests/test_encode_decode.rs`
- [X] T017 [P] [US5] Create unit test for special tokens in `crates/nanochat-tokenizer/tests/test_special_tokens.rs`
- [X] T018 [P] [US5] Create property-based test for encode/decode round-trip in `crates/nanochat-tokenizer/tests/test_roundtrip.rs`

### Implementation for Tokenizer

- [X] T019 [US5] Create `crates/nanochat-tokenizer/src/lib.rs` with public API exports
- [X] T020 [US5] Implement `SpecialTokens` struct in `crates/nanochat-tokenizer/src/special_tokens.rs` with BOS, EOS, PAD token definitions
- [X] T021 [US5] Implement `Vocabulary` struct in `crates/nanochat-tokenizer/src/vocab.rs` with token-to-ID and ID-to-token mappings
- [X] T022 [US5] Implement BPE training algorithm in `crates/nanochat-tokenizer/src/bpe.rs` using aprender primitives
- [X] T023 [US5] Implement `encode` method in `crates/nanochat-tokenizer/src/bpe.rs` to convert text to token IDs
- [X] T024 [US5] Implement `decode` method in `crates/nanochat-tokenizer/src/bpe.rs` to convert token IDs to text
- [X] T025 [US5] Implement `Tokenizer` struct in `crates/nanochat-tokenizer/src/lib.rs` combining vocab, BPE, and special tokens
- [X] T026 [US5] Add comprehensive documentation for all public APIs in `crates/nanochat-tokenizer/src/lib.rs`
- [X] T027 [US5] Add error handling for invalid inputs and edge cases in `crates/nanochat-tokenizer/src/lib.rs`
- [X] T028 [US5] Implement tokenizer save/load functionality using aprender's `.apr` format in `crates/nanochat-tokenizer/src/lib.rs`

**Checkpoint**: Tokenizer crate complete, all tests passing, can encode/decode text with 95%+ fidelity

---

## Phase 3: Model Architecture Development (LLMOps: Model Implementation)

**LLMOps Step**: Model Implementation  
**Purpose**: Implement core GPT architecture - required for all training and inference  
**Crate**: `nanochat-model`  
**Dependencies**: Phase 2 (Tokenizer for vocabulary size)

### Tests for Model (Write First)

- [X] T029 [P] Create unit test for ModelConfig validation in `crates/nanochat-model/tests/unit/test_config.rs`
- [X] T030 [P] Create unit test for RMSNorm in `crates/nanochat-model/tests/test_norm.rs`
- [X] T031 [P] Create unit test for RoPE in `crates/nanochat-model/tests/test_rope.rs`
- [X] T032 [P] Create unit test for attention mechanism in `crates/nanochat-model/tests/test_attention.rs`
- [X] T033 [P] Create unit test for MLP with ReLU² in `crates/nanochat-model/tests/test_mlp.rs`
- [X] T034 [P] Create integration test for forward pass in `crates/nanochat-model/tests/test_forward.rs`
- [X] T035 [P] Create integration test for checkpoint save/load in `crates/nanochat-model/tests/test_checkpoint.rs`

### Implementation for Model

- [X] T036 Create `crates/nanochat-model/src/lib.rs` with public API exports
- [X] T037 Implement `ModelConfig` struct in `crates/nanochat-model/src/config.rs` with validation (vocab_size, n_layer, n_head, n_kv_head, n_embd, sequence_len, dropout)
- [X] T038 Implement `RMSNorm` function in `crates/nanochat-model/src/norm.rs` using aprender primitives (no learnable parameters)
- [X] T039 Implement `RotaryPositionEmbedding` in `crates/nanochat-model/src/rope.rs` using aprender's `RotaryPositionEmbedding` with precomputed frequencies
- [X] T040 Implement QK normalization in `crates/nanochat-model/src/attention.rs` using aprender primitives (normalize queries and keys after RoPE)
- [X] T041 Implement `GroupedQueryAttention` in `crates/nanochat-model/src/attention.rs` using aprender's `GroupedQueryAttention` with KV cache support
- [X] T042 Implement causal attention masking in `crates/nanochat-model/src/attention.rs` for autoregressive generation
- [X] T043 Implement MLP layer with ReLU² activation in `crates/nanochat-model/src/mlp.rs` using aprender's `Linear` layers and functional API for ReLU²
- [X] T044 Implement transformer decoder block in `crates/nanochat-model/src/gpt.rs` combining attention, MLP, and RMSNorm with pre-norm residual connections
- [X] T045 Implement token embedding layer in `crates/nanochat-model/src/gpt.rs` with RMSNorm after embedding (untied from LM head)
- [X] T046 Implement language model head in `crates/nanochat-model/src/gpt.rs` (untied from embedding, separate parameters)
- [X] T047 Implement `GPTModel` struct in `crates/nanochat-model/src/gpt.rs` with configurable depth (n_layer transformer blocks)
- [ ] T048 Implement forward pass in `crates/nanochat-model/src/gpt.rs` with KV cache support for inference
- [ ] T049 Implement checkpoint save functionality in `crates/nanochat-model/src/checkpoint.rs` using aprender's `.apr` format (weights, config, metadata)
- [ ] T050 Implement checkpoint load functionality in `crates/nanochat-model/src/checkpoint.rs` with integrity validation (checksum, version)
- [ ] T051 Add comprehensive documentation for all public APIs in `crates/nanochat-model/src/lib.rs`
- [ ] T052 Add numerical stability checks for all mathematical operations (overflow, underflow, NaN detection)
- [ ] T053 Implement benchmark for forward pass performance in `crates/nanochat-model/benches/forward.rs`

**Checkpoint**: Model crate complete, all tests passing, can perform forward pass and save/load checkpoints

---

## Phase 4: Training Pipeline Development - Pretraining (LLMOps: Training Infrastructure)

**LLMOps Step**: Training Infrastructure - Pretraining Stage  
**Purpose**: Implement pretraining stage for base language modeling  
**Crate**: `nanochat-pretrain`  
**Dependencies**: Phase 3 (Model), Phase 2 (Tokenizer)

### Tests for Pretraining (Write First)

- [ ] T054 [P] [US3] Create unit test for data loading in `crates/nanochat-pretrain/tests/test_dataloader.rs`
- [ ] T055 [P] [US3] Create unit test for optimizer configuration in `crates/nanochat-pretrain/tests/test_optimizer.rs`
- [ ] T056 [P] [US3] Create integration test for training loop in `crates/nanochat-pretrain/tests/test_train.rs`

### Implementation for Pretraining

- [ ] T057 [US3] Create `crates/nanochat-pretrain/src/main.rs` with CLI entry point using clap
- [ ] T058 [US3] Implement command-line argument parsing in `crates/nanochat-pretrain/src/main.rs` (config, data-dir, output-dir, resume, device, workers, etc.)
- [ ] T059 [US3] Implement `DataLoader` in `crates/nanochat-pretrain/src/dataloader.rs` with shuffling, batching, and tokenization
- [ ] T060 [US3] Implement data sharding support in `crates/nanochat-pretrain/src/dataloader.rs` for large datasets
- [ ] T061 [US3] Implement gradient accumulation logic in `crates/nanochat-pretrain/src/train.rs` for effective larger batch sizes
- [ ] T062 [US3] Implement AdamW optimizer integration in `crates/nanochat-pretrain/src/optimizer.rs` using aprender's `AdamW` with configurable hyperparameters
- [ ] T063 [US3] Implement learning rate scheduling in `crates/nanochat-pretrain/src/optimizer.rs` using aprender's `WarmupCosineScheduler`
- [ ] T064 [US3] Implement training loop in `crates/nanochat-pretrain/src/train.rs` with forward pass, backward pass, and optimizer step
- [ ] T065 [US3] Implement checkpoint saving at intervals in `crates/nanochat-pretrain/src/train.rs` using model's checkpoint functionality
- [ ] T066 [US3] Implement checkpoint resumption in `crates/nanochat-pretrain/src/train.rs` to resume from saved checkpoints
- [ ] T067 [US3] Implement training metrics logging in `crates/nanochat-pretrain/src/metrics.rs` (loss, learning rate, throughput)
- [ ] T068 [US3] Add comprehensive documentation for CLI interface and training process in `crates/nanochat-pretrain/src/main.rs`
- [ ] T069 [US3] Add error handling for training failures and checkpoint errors in `crates/nanochat-pretrain/src/train.rs`

**Checkpoint**: Pretraining crate complete, can train base model and save checkpoints

---

## Phase 5: Training Pipeline Development - Mid-Training (LLMOps: Training Infrastructure)

**LLMOps Step**: Training Infrastructure - Mid-Training Stage  
**Purpose**: Implement mid-training stage for conversational fine-tuning  
**Crate**: `nanochat-midtrain`  
**Dependencies**: Phase 4 (Pretraining - for base model checkpoint)

### Tests for Mid-Training (Write First)

- [ ] T070 [P] [US3] Create unit test for conversational data loading in `crates/nanochat-midtrain/tests/test_dataloader.rs`
- [ ] T071 [P] [US3] Create integration test for mid-training loop in `crates/nanochat-midtrain/tests/test_train.rs`

### Implementation for Mid-Training

- [ ] T072 [US3] Create `crates/nanochat-midtrain/src/main.rs` with CLI entry point using clap
- [ ] T073 [US3] Implement command-line argument parsing in `crates/nanochat-midtrain/src/main.rs` (config, base-model, data-dir, output-dir, resume, device, workers, etc.)
- [ ] T074 [US3] Implement conversational data loading in `crates/nanochat-midtrain/src/dataloader.rs` with conversation format support
- [ ] T075 [US3] Implement training loop in `crates/nanochat-midtrain/src/train.rs` reusing pretraining infrastructure but with conversational data
- [ ] T076 [US3] Implement checkpoint saving and resumption in `crates/nanochat-midtrain/src/train.rs`
- [ ] T077 [US3] Implement training metrics logging in `crates/nanochat-midtrain/src/train.rs`
- [ ] T078 [US3] Add comprehensive documentation for CLI interface in `crates/nanochat-midtrain/src/main.rs`
- [ ] T079 [US3] Add error handling for training failures in `crates/nanochat-midtrain/src/train.rs`

**Checkpoint**: Mid-training crate complete, can fine-tune pretrained model for conversational ability

---

## Phase 6: Training Pipeline Development - Supervised Fine-Tuning (LLMOps: Training Infrastructure)

**LLMOps Step**: Training Infrastructure - SFT Stage  
**Purpose**: Implement supervised fine-tuning stage for instruction following  
**Crate**: `nanochat-sft`  
**Dependencies**: Phase 5 (Mid-Training - for mid-trained model checkpoint)

### Tests for SFT (Write First)

- [ ] T080 [P] [US3] Create unit test for instruction data loading in `crates/nanochat-sft/tests/test_dataloader.rs`
- [ ] T081 [P] [US3] Create integration test for SFT training loop in `crates/nanochat-sft/tests/test_train.rs`

### Implementation for SFT

- [ ] T082 [US3] Create `crates/nanochat-sft/src/main.rs` with CLI entry point using clap
- [ ] T083 [US3] Implement command-line argument parsing in `crates/nanochat-sft/src/main.rs` (config, base-model, data-dir, output-dir, resume, device, workers, etc.)
- [ ] T084 [US3] Implement instruction-following data loading in `crates/nanochat-sft/src/dataloader.rs` with instruction-response format support
- [ ] T085 [US3] Implement training loop in `crates/nanochat-sft/src/train.rs` reusing training infrastructure but with instruction data
- [ ] T086 [US3] Implement checkpoint saving and resumption in `crates/nanochat-sft/src/train.rs`
- [ ] T087 [US3] Implement training metrics logging in `crates/nanochat-sft/src/train.rs`
- [ ] T088 [US3] Add comprehensive documentation for CLI interface in `crates/nanochat-sft/src/main.rs`
- [ ] T089 [US3] Add error handling for training failures in `crates/nanochat-sft/src/train.rs`

**Checkpoint**: SFT crate complete, can fine-tune mid-trained model for instruction following

---

## Phase 7: Evaluation & Validation (LLMOps: Model Evaluation & Benchmarking)

**LLMOps Step**: Model Evaluation & Benchmarking  
**Purpose**: Implement evaluation on standard benchmarks to assess model quality  
**Crate**: `nanochat-eval`  
**Dependencies**: Phase 3 (Model), Phase 2 (Tokenizer)

### Tests for Evaluation (Write First)

- [ ] T090 [P] [US4] Create unit test for CORE benchmark in `crates/nanochat-eval/tests/test_core.rs`
- [ ] T091 [P] [US4] Create unit test for ARC benchmark in `crates/nanochat-eval/tests/test_arc.rs`
- [ ] T092 [P] [US4] Create unit test for GSM8K benchmark in `crates/nanochat-eval/tests/test_gsm8k.rs`
- [ ] T093 [P] [US4] Create unit test for report generation in `crates/nanochat-eval/tests/test_report.rs`

### Implementation for Evaluation

- [ ] T094 [US4] Create `crates/nanochat-eval/src/lib.rs` with public API exports
- [ ] T095 [US4] Implement CORE benchmark evaluation in `crates/nanochat-eval/src/core.rs` following reference implementation
- [ ] T096 [US4] Implement ARC (Easy and Challenge) benchmark evaluation in `crates/nanochat-eval/src/arc.rs`
- [ ] T097 [US4] Implement GSM8K math reasoning benchmark evaluation in `crates/nanochat-eval/src/gsm8k.rs`
- [ ] T098 [US4] Implement HumanEval code generation benchmark evaluation in `crates/nanochat-eval/src/humaneval.rs`
- [ ] T099 [US4] Implement MMLU knowledge benchmark evaluation in `crates/nanochat-eval/src/mmlu.rs`
- [ ] T100 [US4] Implement ChatCORE conversational benchmark evaluation in `crates/nanochat-eval/src/chatcore.rs`
- [ ] T101 [US4] Implement evaluation report generation in `crates/nanochat-eval/src/report.rs` with scores and metrics
- [ ] T102 [US4] Create `crates/nanochat-eval/src/main.rs` with CLI entry point using clap
- [ ] T103 [US4] Implement command-line argument parsing in `crates/nanochat-eval/src/main.rs` (model, benchmarks, output-dir, device, batch-size, etc.)
- [ ] T104 [US4] Add comprehensive documentation for all benchmarks and CLI interface in `crates/nanochat-eval/src/lib.rs`
- [ ] T105 [US4] Add error handling for benchmark execution failures in `crates/nanochat-eval/src/lib.rs`

**Checkpoint**: Evaluation crate complete, can evaluate models on all standard benchmarks and generate reports

---

## Phase 8: Deployment & Serving - CLI Interface (LLMOps: Model Deployment & Inference)

**LLMOps Step**: Model Deployment & Inference - CLI Interface  
**Purpose**: Implement CLI interface for text generation with real-time progress  
**Crate**: `nanochat-cli`  
**Dependencies**: Phase 3 (Model), Phase 2 (Tokenizer)  
**User Story**: US1 - Text Generation via Command Line (P1)

### Tests for CLI (Write First)

- [ ] T106 [P] [US1] Create integration test for inference command in `crates/nanochat-cli/tests/test_infer.rs`
- [ ] T107 [P] [US1] Create integration test for streaming output in `crates/nanochat-cli/tests/test_streaming.rs`

### Implementation for CLI

- [ ] T108 [US1] Create `crates/nanochat-cli/src/main.rs` with CLI entry point using clap
- [ ] T109 [US1] Implement `infer` command in `crates/nanochat-cli/src/commands/infer.rs` with argument parsing (model, prompt, temperature, top-k, top-p, max-tokens, seed, device, stream)
- [ ] T110 [US1] Implement model loading in `crates/nanochat-cli/src/commands/infer.rs` using model crate's checkpoint loading
- [ ] T111 [US1] Implement tokenizer loading in `crates/nanochat-cli/src/commands/infer.rs` using tokenizer crate
- [ ] T112 [US1] Implement text generation loop in `crates/nanochat-cli/src/commands/infer.rs` with KV cache for autoregressive generation
- [ ] T113 [US1] Implement sampling strategies in `crates/nanochat-cli/src/commands/infer.rs` (greedy, temperature, top-k, top-p, combined)
- [ ] T114 [US1] Implement streaming text output in `crates/nanochat-cli/src/commands/infer.rs` with token-by-token generation
- [ ] T115 [US1] Implement progress bars using indicatif in `crates/nanochat-cli/src/ui.rs` for real-time generation feedback
- [ ] T116 [US1] Implement context window truncation handling in `crates/nanochat-cli/src/commands/infer.rs` for prompts exceeding max length
- [ ] T117 [US1] Add comprehensive documentation for CLI interface in `crates/nanochat-cli/src/main.rs`
- [ ] T118 [US1] Add error handling for model loading, tokenization, and generation errors in `crates/nanochat-cli/src/commands/infer.rs`

**Checkpoint**: CLI crate complete, can generate text from command line with streaming and progress indicators

---

## Phase 9: Deployment & Serving - Inference Server (LLMOps: Model Deployment & Inference)

**LLMOps Step**: Model Deployment & Inference - Web Server  
**Purpose**: Implement OpenAI-compatible inference server with streaming support  
**Crate**: `nanochat-inference`  
**Dependencies**: Phase 3 (Model), Phase 2 (Tokenizer), Phase 8 (CLI - for inference logic reuse)  
**User Story**: US2 - Interactive Web Chat Interface (P1)

### Tests for Inference Server (Write First)

- [ ] T119 [P] [US2] Create contract test for `/v1/chat/completions` endpoint in `crates/nanochat-inference/tests/contract/test_chat_completions.rs`
- [ ] T120 [P] [US2] Create contract test for `/v1/models` endpoint in `crates/nanochat-inference/tests/contract/test_models.rs`
- [ ] T121 [P] [US2] Create integration test for streaming responses in `crates/nanochat-inference/tests/integration/test_streaming.rs`
- [ ] T122 [P] [US2] Create integration test for conversation session management in `crates/nanochat-inference/tests/integration/test_session.rs`

### Implementation for Inference Server

- [ ] T123 [US2] Create `crates/nanochat-inference/src/main.rs` with server entry point
- [ ] T124 [US2] Implement actix-web server setup in `crates/nanochat-inference/src/server.rs` with configurable host, port, and workers
- [ ] T125 [US2] Implement model loading and management in `crates/nanochat-inference/src/server.rs` with support for multiple model replicas
- [ ] T126 [US2] Implement OpenAI-compatible request parsing in `crates/nanochat-inference/src/handlers.rs` (model, messages, temperature, top_p, max_tokens, stream, etc.)
- [ ] T127 [US2] Implement OpenAI-compatible non-streaming response format in `crates/nanochat-inference/src/handlers.rs` (id, object, created, model, choices, usage)
- [ ] T128 [US2] Implement OpenAI-compatible streaming response format in `crates/nanochat-inference/src/streaming.rs` (chat.completion.chunk, delta objects, finish_reason)
- [ ] T129 [US2] Implement Server-Sent Events (SSE) streaming in `crates/nanochat-inference/src/streaming.rs` using actix-web
- [ ] T130 [US2] Implement conversation session management in `crates/nanochat-inference/src/session.rs` with message history and context window tracking
- [ ] T131 [US2] Implement inference caching in `crates/nanochat-inference/src/cache.rs` with time-based and size-based invalidation strategies
- [ ] T132 [US2] Implement `/v1/chat/completions` POST endpoint handler in `crates/nanochat-inference/src/handlers.rs`
- [ ] T133 [US2] Implement `/v1/models` GET endpoint handler in `crates/nanochat-inference/src/handlers.rs` returning available models
- [ ] T134 [US2] Implement `/health` GET endpoint handler in `crates/nanochat-inference/src/handlers.rs` for health checks
- [ ] T135 [US2] Implement `/stats` GET endpoint handler in `crates/nanochat-inference/src/handlers.rs` for server statistics
- [ ] T136 [US2] Implement `/cache/invalidate` POST endpoint handler in `crates/nanochat-inference/src/handlers.rs`
- [ ] T137 [US2] Implement OpenAI-compatible error format in `crates/nanochat-inference/src/handlers.rs` (error.message, error.type, error.code)
- [ ] T138 [US2] Implement concurrent request handling using tokio async runtime in `crates/nanochat-inference/src/server.rs`
- [ ] T139 [US2] Add comprehensive documentation for API endpoints in `crates/nanochat-inference/src/handlers.rs`
- [ ] T140 [US2] Add error handling for all API endpoints in `crates/nanochat-inference/src/handlers.rs`

**Checkpoint**: Inference server crate complete, can serve OpenAI-compatible API with streaming support

---

## Phase 10: Integration & Testing (LLMOps: End-to-End Validation)

**LLMOps Step**: End-to-End Validation  
**Purpose**: Comprehensive integration testing, performance validation, and documentation  
**Crate**: Workspace-level integration tests  
**Dependencies**: All previous phases

### Integration Tests

- [ ] T141 [P] Create integration test for complete training pipeline (pretrain → midtrain → sft) in `tests/integration/test_training_pipeline.rs`
- [ ] T142 [P] Create integration test for inference workflow (load model → generate text) in `tests/integration/test_inference_workflow.rs`
- [ ] T143 [P] Create integration test for CLI inference with streaming in `tests/integration/test_cli_streaming.rs`
- [ ] T144 [P] Create integration test for web server inference with streaming in `tests/integration/test_web_streaming.rs`
- [ ] T145 [P] Create integration test for checkpoint save/load round-trip in `tests/integration/test_checkpoint_roundtrip.rs`
- [ ] T146 [P] Create integration test for numerical parity with reference implementation in `tests/integration/test_numerical_parity.rs`

### Performance Benchmarks

- [ ] T147 [P] Create benchmark for model forward pass performance in `tests/benchmarks/bench_forward.rs`
- [ ] T148 [P] Create benchmark for inference throughput in `tests/benchmarks/bench_inference.rs`
- [ ] T149 [P] Create benchmark for training throughput in `tests/benchmarks/bench_training.rs`
- [ ] T150 [P] Create benchmark for tokenizer encoding/decoding performance in `tests/benchmarks/bench_tokenizer.rs`

### Validation & Documentation

- [ ] T151 Validate 100% test coverage for all public APIs using coverage tools
- [ ] T152 Validate numerical parity with reference implementation (within tolerance: 1e-4 for logits, 1e-6 for outputs)
- [ ] T153 Validate OpenAI API compatibility by testing against OpenAI-compatible client libraries
- [ ] T154 Create comprehensive API documentation in `docs/api.md` with examples
- [ ] T155 Update quickstart.md with actual working examples from implementation
- [ ] T156 Create architecture documentation in `docs/architecture.md` explaining crate structure and data flow
- [ ] T157 Create deployment guide in `docs/deployment.md` with production deployment recommendations
- [ ] T158 Run load testing for concurrent users (target: 10+ simultaneous users) and document results

**Checkpoint**: All integration tests passing, performance benchmarks meeting targets, documentation complete

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - can start immediately
- **Phase 2 (Tokenizer)**: Depends on Phase 1 - BLOCKS all ML operations
- **Phase 3 (Model)**: Depends on Phase 2 (for vocab_size) - BLOCKS all training and inference
- **Phase 4 (Pretraining)**: Depends on Phase 3 (Model) and Phase 2 (Tokenizer)
- **Phase 5 (Mid-Training)**: Depends on Phase 4 (Pretraining checkpoint)
- **Phase 6 (SFT)**: Depends on Phase 5 (Mid-Training checkpoint)
- **Phase 7 (Evaluation)**: Depends on Phase 3 (Model) and Phase 2 (Tokenizer) - can run in parallel with training phases if model available
- **Phase 8 (CLI)**: Depends on Phase 3 (Model) and Phase 2 (Tokenizer) - can run in parallel with training phases
- **Phase 9 (Inference Server)**: Depends on Phase 3 (Model), Phase 2 (Tokenizer), and Phase 8 (CLI inference logic)
- **Phase 10 (Integration)**: Depends on all previous phases

### LLMOps Step Mapping

- **Phase 1**: Project Setup & Infrastructure
- **Phase 2**: Data Pipeline Development (Tokenizer)
- **Phase 3**: Model Implementation (GPT Architecture)
- **Phase 4-6**: Training Infrastructure (Pretraining, Mid-Training, SFT)
- **Phase 7**: Model Evaluation & Benchmarking
- **Phase 8-9**: Model Deployment & Inference (CLI and Web Server)
- **Phase 10**: End-to-End Validation

### Parallel Opportunities

- **Phase 1**: All crate Cargo.toml creation tasks [P] can run in parallel
- **Phase 2**: All tokenizer test tasks [P] can run in parallel
- **Phase 3**: All model test tasks [P] can run in parallel
- **Phase 4-6**: Training crates can be developed in parallel after Phase 3 (but execution is sequential: pretrain → midtrain → sft)
- **Phase 7-8**: Evaluation and CLI can run in parallel after Phase 3
- **Phase 9**: Can start after Phase 8 (but can reuse CLI inference logic)
- **Phase 10**: All integration tests [P] can run in parallel

### User Story Dependencies

- **User Story 1 (CLI Inference - P1)**: Phase 8 - Depends on Phase 3 (Model) and Phase 2 (Tokenizer)
- **User Story 2 (Web Interface - P1)**: Phase 9 - Depends on Phase 3, Phase 2, and Phase 8
- **User Story 3 (Training Pipeline - P2)**: Phases 4-6 - Depends on Phase 3 and Phase 2
- **User Story 4 (Evaluation - P2)**: Phase 7 - Depends on Phase 3 and Phase 2
- **User Story 5 (Tokenizer - P3)**: Phase 2 - No dependencies (foundational)

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 - P1)

1. Complete Phase 1: Setup
2. Complete Phase 2: Tokenizer (foundational)
3. Complete Phase 3: Model (foundational)
4. Complete Phase 8: CLI Inference (User Story 1)
5. Complete Phase 9: Inference Server (User Story 2)
6. **STOP and VALIDATE**: Test both user stories independently
7. Deploy/demo if ready

### Incremental Delivery

1. **MVP**: Setup + Tokenizer + Model + CLI + Inference Server (Phases 1-3, 8-9)
   - Delivers: Working inference via CLI and web interface
   - Test independently
   - Deploy/Demo

2. **Training Capability**: Add Pretraining + Mid-Training + SFT (Phases 4-6)
   - Delivers: Complete training pipeline
   - Test independently
   - Deploy/Demo

3. **Evaluation Capability**: Add Evaluation (Phase 7)
   - Delivers: Benchmark evaluation
   - Test independently
   - Deploy/Demo

4. **Production Ready**: Add Integration & Testing (Phase 10)
   - Delivers: Full validation and documentation
   - Production deployment

### Parallel Team Strategy

With multiple developers:

1. **Team completes Phases 1-3 together** (Setup, Tokenizer, Model - foundational)
2. **Once Phase 3 is done**:
   - Developer A: Phase 8 (CLI) + Phase 9 (Inference Server) - User Stories 1 & 2
   - Developer B: Phase 4 (Pretraining) - User Story 3
   - Developer C: Phase 7 (Evaluation) - User Story 4
3. **After Phase 4**:
   - Developer B: Phase 5 (Mid-Training) - User Story 3
4. **After Phase 5**:
   - Developer B: Phase 6 (SFT) - User Story 3
5. **All phases complete**:
   - Team: Phase 10 (Integration & Testing)

---

## Notes

- [P] tasks = different files, no dependencies within the same phase
- [Story] label maps task to specific user story for traceability
- Each phase corresponds to a single LLMOps step and focuses on a single crate (or related crates)
- Phases are sequential - each depends only on previous phases
- Each phase should be independently testable and production-ready before moving to next phase
- All tasks include exact file paths for immediate execution
- Comprehensive documentation required for all public APIs
- 100% test coverage required for all public APIs
- All code must compile without warnings and pass clippy lints

