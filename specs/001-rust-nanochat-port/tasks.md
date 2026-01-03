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

- [X] T019 [US5] Create `crates/nanochat-tokenizer/src/lib.rs` with public API exports wrapping `aprender::text::tokenize::BpeTokenizer` (per Principle VII, FR-026)
- [X] T020 [US5] Integrate special tokens handling via `aprender::text::tokenize::BpeTokenizer` - special tokens (BOS: `<|bos|>`, conversational: `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, optional tool call tokens) are handled by aprender's BPE tokenizer implementation in `crates/nanochat-tokenizer/src/lib.rs` (per Principle VII, no custom struct needed)
- [X] T021 [US5] Integrate vocabulary management via `aprender::text::tokenize::BpeTokenizer` - vocabulary (token-to-ID and ID-to-token mappings) is provided by aprender's BPE tokenizer via `vocab()` and `merges()` methods in `crates/nanochat-tokenizer/src/lib.rs` (per Principle VII)
- [X] T022 [US5] Integrate BPE training via `aprender::text::tokenize::BpeTokenizer::train()` - BPE training algorithm is provided by aprender's BPE tokenizer, called through `Tokenizer::train_from_iterator()` in `crates/nanochat-tokenizer/src/lib.rs` (per Principle VII, no custom implementation)
- [X] T023 [US5] Integrate encoding via `aprender::text::tokenize::BpeTokenizer::encode()` - text-to-token-ID encoding is provided by aprender's BPE tokenizer, exposed through `Tokenizer::encode()` in `crates/nanochat-tokenizer/src/lib.rs` (per Principle VII)
- [X] T024 [US5] Integrate decoding via `aprender::text::tokenize::BpeTokenizer::decode()` - token-ID-to-text decoding is provided by aprender's BPE tokenizer, exposed through `Tokenizer::decode()` in `crates/nanochat-tokenizer/src/lib.rs` (per Principle VII)
- [X] T025 [US5] Implement `Tokenizer` struct in `crates/nanochat-tokenizer/src/lib.rs` wrapping `aprender::text::tokenize::BpeTokenizer` (per Principle VII, no custom BPE implementation)
- [X] T026 [US5] Add comprehensive documentation for all public APIs in `crates/nanochat-tokenizer/src/lib.rs`
- [X] T027 [US5] Add error handling for invalid inputs and edge cases in `crates/nanochat-tokenizer/src/lib.rs`
- [X] T027.1 [US5] Implement `TokenizerData` struct in `crates/nanochat-tokenizer/src/lib.rs` for JSON serialization (vocabulary and merges only, per FR-084, FR-085)
- [X] T028 [US5] Implement tokenizer save/load functionality using JSON format (`tokenizer.json`) in `crates/nanochat-tokenizer/src/lib.rs` - serializes `TokenizerData` struct (vocabulary and merges) to compact JSON, matching Python reference implementation. Note: SafeTensors format is used for model checkpoints only, not tokenizers

**Checkpoint**: Tokenizer crate complete, all tests passing, can encode/decode text with 95%+ fidelity

**Constitution Compliance Verification** (per REMEDIATION_PLAN.md Phase 6):
- [X] C001 Verify all ML operations use aprender APIs (Principle VII) - Tokenizer uses `aprender::text::tokenize::BpeTokenizer` directly
- [X] C002 Verify no custom RNG implementations (Principle VII) - N/A for tokenizer
- [X] C003 Verify weight initialization uses aprender patterns (Principle VII) - N/A for tokenizer
- [X] C004 Verify dropout uses aprender's Dropout (Principle VII) - N/A for tokenizer
- [X] C005 Verify no C/C++ dependencies (Principle I) - All dependencies are pure Rust
- [X] C006 Verify all public APIs have doc comments (Principle V) - All public APIs documented
- [X] C007 Verify code compiles without warnings (Development Workflow) - Verified via quality gates

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
- [X] T038 Implement `RMSNorm` function in `crates/nanochat-model/src/norm.rs` using `aprender::nn::RMSNorm` (per Principle VII, FR-004, FR-086 - no learnable parameters)
- [X] T039 Implement `RotaryPositionEmbedding` in `crates/nanochat-model/src/rope.rs` using aprender's `RotaryPositionEmbedding` with precomputed frequencies
- [X] T040 Implement QK normalization in `crates/nanochat-model/src/attention.rs` using aprender primitives (normalize queries and keys after RoPE)
- [X] T041 Implement `GroupedQueryAttention` in `crates/nanochat-model/src/attention.rs` using aprender's `GroupedQueryAttention` with KV cache support
- [X] T042 Implement causal attention masking in `crates/nanochat-model/src/attention.rs` for autoregressive generation
- [X] T043 Implement MLP layer with ReLU² activation in `crates/nanochat-model/src/mlp.rs` using aprender's `Linear` layers and functional API for ReLU²
- [X] T044 Implement transformer decoder block in `crates/nanochat-model/src/gpt.rs` combining attention, MLP, and RMSNorm with pre-norm residual connections
- [X] T045 Implement token embedding layer in `crates/nanochat-model/src/gpt.rs` with RMSNorm after embedding (untied from LM head)
- [X] T046 Implement language model head in `crates/nanochat-model/src/gpt.rs` (untied from embedding, separate parameters)
- [X] T047 Implement `GPTModel` struct in `crates/nanochat-model/src/gpt.rs` with configurable depth (n_layer transformer blocks)
- [X] T048 Implement forward pass in `crates/nanochat-model/src/gpt.rs` with KV cache support for inference
- [X] T049 Implement checkpoint save functionality in `crates/nanochat-model/src/checkpoint.rs` using SafeTensors format (`.safetensors` file for weights via `aprender::nn::serialize::save_model()`, JSON for metadata) - per plan.md and FR-045
- [X] T050 Implement checkpoint load functionality in `crates/nanochat-model/src/checkpoint.rs` with integrity validation (SafeTensors format validation via `aprender::nn::serialize::load_model()`, version metadata in JSON) - per plan.md and FR-046
- [X] T051 Add comprehensive documentation for all public APIs in `crates/nanochat-model/src/lib.rs`
- [X] T052 Add numerical stability checks for all mathematical operations (overflow, underflow, NaN detection) in `crates/nanochat-model/src/stability.rs` using `Tensor::data()` for checks
- [X] T053 Implement benchmark for forward pass performance in `crates/nanochat-model/benches/forward.rs`
- [X] T053.1 [P] Verify 100% test coverage for all public APIs in `nanochat-model` crate using coverage tools
- [X] T053.2 [P] Verify 100% test coverage for all public APIs in `nanochat-tokenizer` crate using coverage tools
- [X] T053.3 [P] Run `cargo test --workspace --all-features` and verify all tests pass (per FR-088 quality gates)
- [X] T053.4 [P] Run `cargo clippy --workspace --all-features --all-targets` and verify no warnings (per FR-088 quality gates)
- [X] T053.5 [P] Run `cargo fmt --all` and verify all code is formatted (per FR-088 quality gates)
- [X] T053.6 [P] Verify all tests use `.expect()` instead of `.unwrap()` (per FR-089, constitution requirement)

**Checkpoint**: Model crate complete, all tests passing, can perform forward pass and save/load checkpoints

**Constitution Compliance Verification** (per REMEDIATION_PLAN.md Phase 6):
- [X] C008 Verify all ML operations use aprender APIs (Principle VII) - Uses `aprender::nn::RMSNorm`, `aprender::nn::init`, `aprender::nn::Dropout`, `aprender::nn::loss::CrossEntropyLoss`, `aprender::nn::serialize`
- [X] C009 Verify no custom RNG implementations (Principle VII) - Uses `rand::rngs::StdRng` with proper seeding
- [X] C010 Verify weight initialization uses aprender patterns (Principle VII) - Uses `aprender::nn::init::normal()` via `init_linear_weight()` helper
- [X] C011 Verify dropout uses aprender's Dropout (Principle VII) - Uses `aprender::nn::Dropout` with proper seeding
- [X] C012 Verify loss computation uses aprender APIs (Principle VII) - Uses `aprender::nn::loss::CrossEntropyLoss` (FR-087)
- [X] C013 Verify checkpoint serialization uses aprender APIs (Principle VII) - Uses `aprender::nn::serialize::{save_model, load_model}` with SafeTensors format
- [X] C014 Verify no C/C++ dependencies (Principle I) - All dependencies are pure Rust
- [X] C015 Verify all public APIs have doc comments (Principle V) - All public APIs documented
- [X] C016 Verify code compiles without warnings (Development Workflow) - Verified via quality gates

**Architecture Notes**:
- Model crate provides `forward_cache()` method for inference with KV cache support
- `generate()` method is NOT in model crate - implemented in inference crates (CLI, inference server) per plan.md
- `setup_optimizers()` method is NOT in model crate - implemented in training crates per plan.md
- Device selection is compile-time via `gpu` feature flag, not runtime (no `get_device()` method)

---

## Phase 4: Training Pipeline Development - Pretraining (LLMOps: Training Infrastructure)

**LLMOps Step**: Training Infrastructure - Pretraining Stage  
**Purpose**: Implement pretraining stage for base language modeling  
**Crate**: `nanochat-pretrain`  
**Dependencies**: Phase 3 (Model), Phase 2 (Tokenizer)

### Tests for Pretraining (Write First)

- [X] T054 [P] [US3] Create unit test for data loading in `crates/nanochat-pretrain/tests/test_dataloader.rs`
- [X] T055 [P] [US3] Create unit test for optimizer configuration in `crates/nanochat-pretrain/tests/test_optimizer.rs`
- [X] T056 [P] [US3] Create integration test for training loop in `crates/nanochat-pretrain/tests/test_train.rs`

### Implementation for Pretraining

- [X] T057 [US3] Create `crates/nanochat-pretrain/src/main.rs` with CLI entry point using clap
- [X] T058 [US3] Implement command-line argument parsing in `crates/nanochat-pretrain/src/main.rs` (config, data-dir, output-dir, resume, workers, etc.) - Note: No `--device` option (device selection is compile-time via `gpu` feature flag per plan.md)
- [X] T059 [US3] Implement `DataLoader` in `crates/nanochat-pretrain/src/dataloader.rs` with shuffling, batching, and tokenization
- [X] T060 [US3] Implement data sharding support in `crates/nanochat-pretrain/src/dataloader.rs` for large datasets
- [X] T061 [US3] Implement gradient accumulation logic in `crates/nanochat-pretrain/src/train.rs` for effective larger batch sizes
- [X] T062 [US3] Implement AdamW optimizer integration in `crates/nanochat-pretrain/src/optimizer.rs` using `aprender::nn::optim::AdamW` with configurable hyperparameters (per Principle VII, FR-017, FR-086) - Note: This implements the `setup_optimizers()` functionality from Python reference, using model's `parameters_mut()` method to get mutable parameter references for optimizer
- [X] T063 [US3] Implement learning rate scheduling in `crates/nanochat-pretrain/src/optimizer.rs` using `aprender::nn::scheduler::WarmupCosineScheduler` (per Principle VII, FR-024, FR-086)
- [X] T064 [US3] Implement training loop in `crates/nanochat-pretrain/src/train.rs` with forward pass (using model's `forward_training()` with `aprender::nn::loss::CrossEntropyLoss` per FR-087, FR-086), backward pass (using `loss.backward()`), and optimizer step (using `optimizer.step()` and `optimizer.zero_grad()`)
- [X] T065 [US3] Implement checkpoint saving at intervals in `crates/nanochat-pretrain/src/train.rs` using model's checkpoint functionality (SafeTensors format per FR-045, FR-046)
- [X] T066 [US3] Implement checkpoint resumption in `crates/nanochat-pretrain/src/train.rs` to resume from saved checkpoints (SafeTensors format per FR-020, FR-046)
- [X] T067 [US3] Implement training metrics logging in `crates/nanochat-pretrain/src/metrics.rs` (loss, learning rate, throughput) per FR-025
- [X] T068 [US3] Add comprehensive documentation for CLI interface and training process in `crates/nanochat-pretrain/src/main.rs`
- [X] T069 [US3] Add error handling for training failures and checkpoint errors in `crates/nanochat-pretrain/src/train.rs`
- [X] T069.1 [US3] [P] Run quality gates before marking pretraining complete: `cargo fmt --all`, `cargo clippy --workspace --all-features --all-targets`, `cargo test --workspace --all-features` (per FR-088)

**Checkpoint**: Pretraining crate complete, can train base model and save checkpoints

---

## Phase 5: Training Pipeline Development - Mid-Training (LLMOps: Training Infrastructure)

**LLMOps Step**: Training Infrastructure - Mid-Training Stage  
**Purpose**: Implement mid-training stage for conversational fine-tuning  
**Crate**: `nanochat-midtrain`  
**Dependencies**: Phase 4 (Pretraining - for base model checkpoint)

### Tests for Mid-Training (Write First)

- [X] T070 [P] [US3] Create unit test for conversational data loading in `crates/nanochat-midtrain/tests/test_dataloader.rs`
- [X] T071 [P] [US3] Create integration test for mid-training loop in `crates/nanochat-midtrain/tests/test_train.rs`

### Implementation for Mid-Training

- [X] T072 [US3] Create `crates/nanochat-midtrain/src/main.rs` with CLI entry point using clap
- [X] T073 [US3] Implement command-line argument parsing in `crates/nanochat-midtrain/src/main.rs` (config, base-model, data-dir, output-dir, resume, workers, etc.) - Note: No `--device` option (device selection is compile-time via `gpu` feature flag per plan.md)
- [X] T074 [US3] Implement conversational data loading in `crates/nanochat-midtrain/src/dataloader.rs` with JSONL conversation format support (messages array with role/content fields, role alternation, system message merging, training mask generation, configurable shuffling with seed for reproducibility)
- [X] T074.1 [US3] Implement conversation tokenization method in `crates/nanochat-tokenizer/src/lib.rs` (equivalent to Python's `render_conversation()`) that handles system message merging, role validation, content type handling (string vs list of parts), and training mask generation
- [X] T075 [US3] Implement training loop in `crates/nanochat-midtrain/src/train.rs` reusing pretraining infrastructure (optimizer setup, learning rate scheduling, gradient accumulation, checkpoint saving/loading, metrics logging) but adapted for conversational data (uses training mask for loss computation, handles conversation tokenization)
- [X] T076 [US3] Implement checkpoint saving and resumption in `crates/nanochat-midtrain/src/train.rs`
- [X] T077 [US3] Implement training metrics logging in `crates/nanochat-midtrain/src/train.rs`
- [X] T078 [US3] Add comprehensive documentation for CLI interface in `crates/nanochat-midtrain/src/main.rs`
- [X] T079 [US3] Add error handling for training failures in `crates/nanochat-midtrain/src/train.rs`

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
- [ ] T083 [US3] Implement command-line argument parsing in `crates/nanochat-sft/src/main.rs` (config, base-model, data-dir, output-dir, resume, workers, etc.) - Note: No `--device` option (device selection is compile-time via `gpu` feature flag per plan.md)
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
- [ ] T103 [US4] Implement command-line argument parsing in `crates/nanochat-eval/src/main.rs` (model, benchmarks, output-dir, batch-size, etc.) - Note: No `--device` option (device selection is compile-time via `gpu` feature flag per plan.md)
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
- [ ] T109 [US1] Implement `infer` command in `crates/nanochat-cli/src/commands/infer.rs` with argument parsing (model, prompt, temperature, top-k, top-p, max-tokens, seed, stream) - Note: No `--device` option (device selection is compile-time via `gpu` feature flag per plan.md). This implements the `generate()` functionality from Python reference using model's `forward_cache()` method
- [ ] T110 [US1] Implement model loading in `crates/nanochat-cli/src/commands/infer.rs` using model crate's checkpoint loading
- [ ] T111 [US1] Implement tokenizer loading in `crates/nanochat-cli/src/commands/infer.rs` using tokenizer crate
- [ ] T112 [US1] Implement text generation loop in `crates/nanochat-cli/src/commands/infer.rs` with KV cache for autoregressive generation - Note: This implements the `generate()` functionality from Python reference, using model's `forward_cache()` method. Model crate provides core forward pass only, CLI crate implements generation and sampling
- [ ] T113 [US1] Implement sampling strategies in `crates/nanochat-cli/src/commands/infer.rs` (greedy, temperature, top-k, top-p, combined)
- [ ] T114 [US1] Implement streaming text output in `crates/nanochat-cli/src/commands/infer.rs` with token-by-token generation
- [ ] T115 [US1] Implement progress bars using indicatif in `crates/nanochat-cli/src/ui.rs` for real-time generation feedback
- [ ] T116 [US1] Implement context window truncation handling in `crates/nanochat-cli/src/commands/infer.rs` for prompts exceeding max length
- [ ] T117 [US1] Add comprehensive documentation for CLI interface in `crates/nanochat-cli/src/main.rs`
- [ ] T118 [US1] Add error handling for model loading, tokenization, and generation errors in `crates/nanochat-cli/src/commands/infer.rs`
- [ ] T118.1 [US1] [P] Run quality gates before marking CLI complete: `cargo fmt --all`, `cargo clippy --workspace --all-features --all-targets`, `cargo test --workspace --all-features` (per FR-088)

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
- [ ] T127 [US2] Implement OpenAI-compatible non-streaming response format in `crates/nanochat-inference/src/handlers.rs` (id, object, created, model, choices, usage) - Note: Uses model's `forward_cache()` method for generation, implementing Python's `generate()` functionality. Model crate provides core forward pass only, inference server implements generation and sampling
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
- [ ] T140.1 [US2] [P] Run quality gates before marking inference server complete: `cargo fmt --all`, `cargo clippy --workspace --all-features --all-targets`, `cargo test --workspace --all-features` (per FR-088)

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

- [ ] T151 Validate 100% test coverage for all public APIs using coverage tools (per FR-065)
- [ ] T152 Validate numerical parity with reference implementation (within tolerance: 1e-4 for logits, 1e-6 for outputs) per FR-070
- [ ] T153 Validate OpenAI API compatibility by testing against OpenAI-compatible client libraries (per FR-080-083, SC-015)
- [ ] T153.1 [P] Run final quality gates across entire workspace: `cargo fmt --all`, `cargo clippy --workspace --all-features --all-targets`, `cargo test --workspace --all-features` (per FR-088)
- [ ] T153.2 [P] Verify all tests use `.expect()` instead of `.unwrap()` across entire workspace (per FR-089, constitution requirement)
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
- Comprehensive documentation required for all public APIs (per FR-071)
- 100% test coverage required for all public APIs (per FR-065)
- All code must compile without warnings and pass clippy lints (per FR-057, FR-074)
- **MANDATORY Quality Gates** (per FR-088, constitution): Before marking any task/phase complete, MUST pass:
  - Formatting: `cargo fmt --all`
  - Linting: `cargo clippy --workspace --all-features --all-targets`
  - Testing: `cargo test --workspace --all-features`
- **Test Error Handling** (per FR-089, constitution): All tests MUST use `.expect("descriptive message")` instead of `.unwrap()` for easier debugging
- **Aprender API Reuse** (per FR-086, Principle VII): All ML operations MUST use aprender APIs - custom implementations that duplicate aprender functionality are FORBIDDEN
- **Format Specifications**: Checkpoints use SafeTensors format (`.safetensors`), tokenizers use JSON format (`tokenizer.json`) per plan.md and spec.md
- **Device Selection**: Compile-time via `gpu` feature flag, not runtime - no `--device` CLI option per plan.md
- **Architecture Deferrals**: `generate()` in inference crates, `setup_optimizers()` in training crates - model crate provides core forward pass only per plan.md

