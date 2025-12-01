# Feature Specification: Production-Grade Rust Port of Nanochat

**Feature Branch**: `001-rust-nanochat-port`  
**Created**: 2025-01-27  
**Status**: Draft  
**Input**: User description: "a full 100% tested and production grade port of nanochat located in @nanochat by Karpathy in Rust programming language taking as examples implementation in @nanochat-rs and @nanogpt-rs but leveraging the new available using the new available of classical machine learning algorithms optimized for performance and safety in pure Rust with zero C/C++ dependencies and strictly align to their code of conduct and practices"

## Clarifications

### Session 2025-01-27

- Q: What ML framework dependencies are allowed? → A: Strictly aprender crate for all ML operations. If aprender does not provide needed functionality, implement using trueno and nalgebra directly. Aprender provides GPU support via `gpu` feature (wgpu backend through trueno) and CPU support. Aprender forks: @aprender, trueno fork: @trueno, nalgebra fork: @nalgebra
- Q: What non-ML dependencies are allowed? → A: For non-ML functionality, use actix-web (web serving), tokio (async runtime), and serde (serialization) to provide robust inference with caching support. For CLI, use clap with indicatif for real-time features including streaming. Architecture: modular crate structure with model as separate crate, each ML cycle step as its own crate, and inference server as its own crate

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Text Generation via Command Line (Priority: P1)

A user wants to generate text using a trained language model through a simple command-line interface. They provide a text prompt and receive generated text continuation with real-time streaming feedback.

**Why this priority**: This is the core value proposition - enabling users to interact with the language model. It demonstrates the complete inference pipeline from tokenization through model forward pass to text generation.

**Independent Test**: Can be fully tested by running a single command with a prompt and verifying that coherent text is generated with real-time progress indicators. This delivers immediate value as a working language model interface.

**Acceptance Scenarios**:

1. **Given** a trained model checkpoint and a text prompt, **When** the user runs the inference command, **Then** the system generates coherent text continuation matching the prompt's style and context, displaying real-time progress via indicatif
2. **Given** a prompt with special characters and unicode, **When** the user runs inference, **Then** the system correctly tokenizes and generates text without errors, with streaming output visible in real-time
3. **Given** a prompt exceeding the maximum context length, **When** the user runs inference, **Then** the system handles truncation gracefully and generates text from the truncated context, showing progress indicators throughout

---

### User Story 2 - Interactive Web Chat Interface (Priority: P1)

A user wants to have a conversational interaction with the language model through a web-based chat interface similar to ChatGPT, with streaming responses and conversation history.

**Why this priority**: The web interface provides the primary user experience and makes the model accessible to non-technical users. It demonstrates the full-stack capability including serving, streaming, and state management.

**Independent Test**: Can be fully tested by starting the web server, opening the interface in a browser, sending messages, and verifying streaming responses appear correctly. This delivers a production-ready user experience.

**Acceptance Scenarios**:

1. **Given** a running web server with a loaded model, **When** a user opens the chat interface and sends a message, **Then** the system streams the response token-by-token in real-time
2. **Given** an active conversation with multiple message exchanges, **When** the user sends a new message, **Then** the system maintains conversation context and generates contextually appropriate responses
3. **Given** multiple concurrent users, **When** they interact with the web interface simultaneously, **Then** the system handles all requests without degradation in response quality or speed

---

### User Story 3 - Model Training Pipeline (Priority: P2)

A researcher or developer wants to train a language model from scratch using the full training pipeline including pretraining, mid-training, and supervised fine-tuning stages.

**Why this priority**: Training capability enables users to create custom models. While inference is the primary use case, training completes the full-stack implementation and demonstrates production-grade ML capabilities.

**Independent Test**: Can be fully tested by running the training pipeline on a small dataset and verifying that model checkpoints are saved, loss decreases over time, and the trained model can generate text. This delivers the ability to create custom language models.

**Acceptance Scenarios**:

1. **Given** a training dataset and configuration, **When** the user runs the pretraining stage, **Then** the system trains the base model, saves periodic checkpoints, and reports training metrics
2. **Given** a pretrained base model, **When** the user runs mid-training, **Then** the system fine-tunes the model for conversational ability and saves the resulting checkpoint
3. **Given** a mid-trained model, **When** the user runs supervised fine-tuning, **Then** the system further refines the model on instruction-following data and produces a chat-capable model

---

### User Story 4 - Model Evaluation and Benchmarking (Priority: P2)

A user wants to evaluate model performance across multiple standard benchmarks to understand model capabilities and compare different training configurations.

**Why this priority**: Evaluation provides objective metrics for model quality and enables comparison between different training runs. It demonstrates production-grade ML practices with comprehensive testing.

**Independent Test**: Can be fully tested by running evaluation on a trained model and verifying that benchmark scores are computed and reported correctly. This delivers quantitative assessment of model quality.

**Acceptance Scenarios**:

1. **Given** a trained model checkpoint, **When** the user runs evaluation on standard benchmarks, **Then** the system computes and reports scores for all benchmark tasks
2. **Given** evaluation results from multiple model checkpoints, **When** the user compares results, **Then** the system provides clear metrics that enable meaningful comparison
3. **Given** a model that fails on a specific benchmark task, **When** the user reviews evaluation output, **Then** the system provides detailed error information to aid debugging

---

### User Story 5 - Tokenizer Training and Management (Priority: P3)

A user wants to train a custom tokenizer on their dataset or use an existing tokenizer for text encoding and decoding operations.

**Why this priority**: Tokenization is a foundational component required for all other operations. While lower priority than inference, it's essential for the complete system.

**Independent Test**: Can be fully tested by training a tokenizer on sample text and verifying that encoding/decoding round-trips correctly preserve the original text. This delivers the text preprocessing capability.

**Acceptance Scenarios**:

1. **Given** a corpus of training text, **When** the user trains a tokenizer, **Then** the system produces a tokenizer that can encode and decode text with high fidelity
2. **Given** a trained tokenizer, **When** the user encodes and decodes text, **Then** the system preserves the original text content (allowing for tokenization artifacts)
3. **Given** text in multiple languages, **When** the user uses the tokenizer, **Then** the system handles multilingual content correctly

---

### Edge Cases

- What happens when model inference runs out of memory on a device with limited resources?
- How does the system handle malformed or corrupted checkpoint files during model loading?
- What happens when a user provides an empty prompt or a prompt containing only whitespace?
- How does the system handle very long conversations that exceed context window limits?
- What happens when training is interrupted (e.g., system crash, user cancellation) - can training resume from the last checkpoint?
- How does the system handle concurrent training and inference requests on the same hardware?
- What happens when tokenizer vocabulary size exceeds available memory?
- How does the system handle numerical overflow or underflow during training or inference?
- What happens when evaluation benchmarks fail due to data format issues?
- How does the system handle network failures during web serving (client disconnects, timeouts)?

## Requirements *(mandatory)*

### Functional Requirements

#### Core Model Architecture

- **FR-001**: System MUST implement a decoder-only transformer architecture with configurable depth (number of layers)
- **FR-002**: System MUST support rotary position embeddings (RoPE) for relative positional encoding
- **FR-003**: System MUST implement QK normalization for attention stability
- **FR-004**: System MUST use RMSNorm for layer normalization without learnable parameters
- **FR-005**: System MUST support Group-Query Attention (GQA) with configurable query and key-value head counts
- **FR-006**: System MUST use ReLU² activation in MLP layers
- **FR-007**: System MUST implement untied weights between token embedding and language model head
- **FR-008**: System MUST apply normalization after token embedding
- **FR-009**: System MUST use bias-free linear layers throughout the architecture

#### Inference and Generation

- **FR-010**: System MUST support efficient inference with KV cache for autoregressive generation
- **FR-011**: System MUST implement multiple sampling strategies: greedy, temperature scaling, top-k, and top-p (nucleus) sampling
- **FR-012**: System MUST support configurable generation parameters (temperature, top-k, top-p, max tokens)
- **FR-013**: System MUST generate text token-by-token with streaming support
- **FR-014**: System MUST handle context windows up to the configured maximum sequence length
- **FR-015**: System MUST support batch inference for processing multiple prompts simultaneously

#### Training Pipeline

- **FR-016**: System MUST support distributed training across multiple devices/nodes
- **FR-017**: System MUST implement AdamW optimizer with configurable hyperparameters
- **FR-018**: System MUST support gradient accumulation for effective larger batch sizes
- **FR-019**: System MUST save training checkpoints at configurable intervals
- **FR-020**: System MUST support resuming training from saved checkpoints
- **FR-021**: System MUST implement pretraining stage for base language modeling
- **FR-022**: System MUST implement mid-training stage for conversational fine-tuning
- **FR-023**: System MUST implement supervised fine-tuning (SFT) stage for instruction following
- **FR-024**: System MUST support learning rate scheduling with warmup and decay
- **FR-025**: System MUST log training metrics (loss, learning rate, throughput) during training

#### Tokenization

- **FR-026**: System MUST implement BPE (Byte Pair Encoding) tokenizer training
- **FR-027**: System MUST support encoding text to token IDs
- **FR-028**: System MUST support decoding token IDs to text
- **FR-029**: System MUST handle special tokens (beginning-of-text, end-of-text, padding, etc.)
- **FR-030**: System MUST support configurable vocabulary size
- **FR-031**: System MUST preserve text fidelity during encode/decode round-trips (within tokenization limitations)

#### Data Management

- **FR-032**: System MUST support loading training data from multiple formats (text files, datasets)
- **FR-033**: System MUST implement efficient data loading with shuffling and batching
- **FR-034**: System MUST support data preprocessing and tokenization during training
- **FR-035**: System MUST handle large datasets that don't fit in memory
- **FR-036**: System MUST support data sharding for distributed training

#### Evaluation and Benchmarking

- **FR-037**: System MUST evaluate models on CORE benchmark for base model quality
- **FR-038**: System MUST evaluate models on ARC (Easy and Challenge) benchmarks
- **FR-039**: System MUST evaluate models on GSM8K math reasoning benchmark
- **FR-040**: System MUST evaluate models on HumanEval code generation benchmark
- **FR-041**: System MUST evaluate models on MMLU knowledge benchmark
- **FR-042**: System MUST evaluate models on ChatCORE conversational benchmark
- **FR-043**: System MUST generate evaluation reports with scores and metrics
- **FR-044**: System MUST support custom evaluation tasks

#### Model Persistence

- **FR-045**: System MUST save model checkpoints including weights and configuration
- **FR-046**: System MUST load model checkpoints for inference or continued training
- **FR-047**: System MUST validate checkpoint integrity during loading
- **FR-048**: System MUST support checkpoint format compatibility across versions

#### Web Interface

- **FR-049**: System MUST provide a web-based chat interface accessible via browser
- **FR-050**: System MUST support Server-Sent Events (SSE) for streaming responses via actix-web with OpenAI-compatible format
- **FR-051**: System MUST maintain conversation history within a session
- **FR-052**: System MUST handle multiple concurrent user sessions with tokio async runtime
- **FR-053**: System MUST provide REST API endpoints for programmatic access with OpenAI-compatible format (`/v1/chat/completions`, `/v1/models`)
- **FR-054**: System MUST support configurable server settings (host, port, workers)
- **FR-080**: System MUST implement OpenAI-compatible request format (model, messages, temperature, top_p, max_tokens, stream, etc.)
- **FR-081**: System MUST implement OpenAI-compatible response format (id, object, created, model, choices, usage) for non-streaming responses
- **FR-082**: System MUST implement OpenAI-compatible streaming format (chat.completion.chunk, delta objects, finish_reason) for streaming responses
- **FR-083**: System MUST support OpenAI-compatible error format (error.message, error.type, error.code)
- **FR-055**: System MUST implement response caching for inference requests to improve performance and reduce redundant computation
- **FR-056**: System MUST support cache invalidation strategies (time-based, size-based, manual) for inference caching

#### Performance and Safety

- **FR-057**: System MUST compile without warnings using standard Rust tooling
- **FR-058**: System MUST have zero C/C++ dependencies (pure Rust implementation)
- **FR-059**: System MUST use aprender crate exclusively for all machine learning operations (tensor operations, neural network layers, optimizers). If aprender does not provide required functionality, System MUST implement using trueno and nalgebra directly
- **FR-060**: System MUST implement numerical stability checks for all mathematical operations
- **FR-061**: System MUST handle memory allocation failures gracefully
- **FR-062**: System MUST prevent buffer overflows and memory safety violations
- **FR-063**: System MUST support GPU acceleration via aprender's `gpu` feature (wgpu backend through trueno) for matrix operations and training
- **FR-064**: System MUST support CPU fallback when GPU is unavailable, using aprender with CPU/SIMD acceleration via trueno

#### Testing and Quality

- **FR-065**: System MUST have 100% test coverage for all public APIs
- **FR-066**: System MUST include unit tests for all core components
- **FR-067**: System MUST include integration tests for end-to-end workflows
- **FR-068**: System MUST include benchmarks for performance-critical paths
- **FR-069**: System MUST pass all tests before code can be merged
- **FR-070**: System MUST maintain numerical parity with reference implementation within acceptable tolerance

#### Code Quality and Documentation

- **FR-071**: System MUST have comprehensive documentation for all public APIs
- **FR-072**: System MUST follow Rust naming conventions and idioms
- **FR-073**: System MUST enforce code formatting via rustfmt
- **FR-074**: System MUST pass clippy lints without warnings
- **FR-075**: System MUST include inline comments explaining complex algorithms
- **FR-076**: System MUST maintain minimal, readable codebase without excessive abstraction

#### Architecture and Modularity

- **FR-077**: System MUST be organized as a modular workspace with separate crates: model crate (core model implementation), individual ML cycle step crates (pretraining, mid-training, SFT, evaluation), and inference server crate
- **FR-078**: System MUST implement CLI interface using clap with indicatif for real-time progress display and streaming output
- **FR-079**: System MUST implement inference server as separate crate using actix-web with tokio for async processing

### Key Entities

- **Model Checkpoint**: Represents a saved state of a trained model, containing model weights, configuration parameters, training metadata (step count, loss values), and version information. Used for resuming training or loading models for inference.

- **Tokenizer**: Represents the text tokenization component that converts between text and token sequences. Contains vocabulary mappings, special token definitions, and encoding/decoding logic. Essential for all text processing operations.

- **Training Configuration**: Represents hyperparameters and settings for model training, including model architecture (depth, width, attention heads), optimizer settings (learning rate, weight decay), batch sizes, and training stage parameters. Determines model behavior and training process.

- **Evaluation Result**: Represents benchmark scores and metrics from model evaluation runs. Contains task names, scores, and metadata. Used for comparing model performance and tracking improvements.

- **Conversation Session**: Represents an active chat interaction with message history, context window state, and generation parameters. Maintains state for multi-turn conversations in the web interface.

- **Training Dataset**: Represents the corpus of text used for training, including data sources, preprocessing steps, and data splits. Provides the learning signal for the language model.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can generate coherent text completions from prompts in under 5 seconds for responses up to 100 tokens on standard hardware (CPU or GPU)

- **SC-002**: The web chat interface streams responses to users with latency under 200ms for the first token and maintains streaming throughput of at least 10 tokens per second

- **SC-003**: The system successfully trains a model from scratch through the complete pipeline (pretraining → mid-training → SFT) and produces a checkpoint that can generate coherent conversational responses

- **SC-004**: Model evaluation completes all standard benchmarks (CORE, ARC, GSM8K, HumanEval, MMLU, ChatCORE) and produces scores within 5% of reference implementation when using identical configurations

- **SC-005**: The system maintains 100% test coverage for all public APIs as measured by code coverage tools, with all tests passing in continuous integration

- **SC-006**: The codebase compiles without warnings and passes all linter checks (rustfmt, clippy) with zero violations

- **SC-007**: The system demonstrates numerical stability with no overflow, underflow, or NaN values during training or inference on standard workloads

- **SC-008**: Model checkpoints can be saved and loaded successfully with 100% weight preservation (bit-exact or within floating-point precision tolerance)

- **SC-009**: The tokenizer achieves text reconstruction fidelity of at least 95% (measured as character-level accuracy) on a diverse test corpus including multilingual and special character content

- **SC-010**: The system handles concurrent requests from at least 10 simultaneous users without degradation in response quality or system stability

- **SC-011**: Training can resume from checkpoints with identical final model quality compared to uninterrupted training runs

- **SC-012**: The system processes training data at a throughput of at least 1000 samples per second per GPU device during pretraining

- **SC-013**: All functional requirements (FR-001 through FR-083) are implemented and verified through automated testing
- **SC-015**: The inference API is fully compatible with OpenAI's Chat Completions API format, enabling drop-in replacement in OpenAI-compatible applications

- **SC-014**: The codebase adheres to all constitution principles (Pure Rust, Performance & Safety, Minimalism, Classical ML Algorithms, Code Quality, LLM Disclosure) as verified through code review

## Project Structure

The implementation MUST be organized as a Cargo workspace with the following crate structure:

```
workspace-root/
├── Cargo.toml                    # Workspace configuration
├── crates/
│   ├── nanochat-model/           # Core model implementation (GPT architecture)
│   ├── nanochat-pretrain/        # Pretraining stage crate
│   ├── nanochat-midtrain/        # Mid-training stage crate
│   ├── nanochat-sft/             # Supervised fine-tuning stage crate
│   ├── nanochat-eval/            # Evaluation and benchmarking crate
│   ├── nanochat-tokenizer/       # Tokenizer implementation crate
│   ├── nanochat-inference/       # Inference server crate (actix-web + tokio)
│   └── nanochat-cli/             # CLI interface crate (clap + indicatif)
└── README.md
```

Each crate MUST be independently testable and have clear, single-purpose responsibilities.

## Assumptions

- Users have access to appropriate hardware (CPU or GPU) for their use case, with GPU recommended for training
- Training datasets are available in standard text formats or can be converted to such formats
- Users are familiar with command-line interfaces for training and inference operations
- Model checkpoints from the reference Python implementation may need conversion utilities (not in scope for this feature)
- The system will be used primarily for inference, with training as a secondary use case
- Standard Rust tooling (cargo, rustc, rustfmt, clippy) is available in the development environment
- The system targets modern Rust (edition 2021 or later) and standard library features
- Inference caching will improve performance for repeated or similar prompts

## Dependencies

### Machine Learning Dependencies

- **aprender crate** (fork: @aprender) - Primary ML framework for all machine learning operations. Provides neural networks, transformers, optimizers, and classical ML algorithms. Uses trueno for SIMD-accelerated operations and nalgebra for linear algebra
- **trueno crate** (fork: @trueno) - High-performance compute library for SIMD operations (CPU) and GPU acceleration via wgpu. Used directly if aprender does not provide needed tensor operations
- **nalgebra crate** (fork: @nalgebra) - Linear algebra library. Used directly if aprender does not provide needed linear algebra operations

### Non-ML Dependencies

- **actix-web** - Web framework for inference server with robust request handling and caching support
- **tokio** - Async runtime for concurrent request processing and streaming
- **serde** - Serialization framework for model checkpoints, API requests/responses, and data formats
- **clap** - Command-line argument parsing for CLI tools
- **indicatif** - Progress bars and real-time display for CLI inference with streaming support

### Development Dependencies

- Standard Rust tooling ecosystem (cargo, rustfmt, clippy)
- Training data in accessible formats
- Hardware resources appropriate for the intended use case (CPU for inference, GPU recommended for training)

## Out of Scope

- Python runtime integration or FFI bindings to Python libraries
- C/C++ dependencies or build-time compilation of C/C++ code
- Model conversion utilities for loading PyTorch checkpoints (may be a separate feature)
- Advanced features not present in the reference nanochat implementation
- Deployment infrastructure and containerization (infrastructure concerns)
- Model serving optimizations beyond basic web interface (advanced serving features)
