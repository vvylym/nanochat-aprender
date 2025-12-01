# Research & Architecture Decisions

**Date**: 2025-01-27  
**Phase**: 0 - Research & Architecture  
**Status**: Complete

## Overview

This document consolidates research findings and architectural decisions for the nanochat Rust port implementation. All decisions align with the constitution principles and LLMOps best practices.

## 1. Aprender Framework Capabilities

### Decision: Use aprender as primary ML framework

**Rationale**: Aprender provides comprehensive neural network support with pure Rust implementation, GPU acceleration via trueno, and transformer components needed for GPT architecture.

**Capabilities Verified**:

1. **Neural Network Modules** (`aprender::nn`):
   - ✅ `Linear` - Fully connected layers (bias-free support)
   - ✅ `RMSNorm` - RMS normalization (no learnable params)
   - ✅ `MultiHeadAttention` - Standard multi-head attention
   - ✅ `GroupedQueryAttention` - GQA support (matches nanochat requirement)
   - ✅ `RotaryPositionEmbedding` - RoPE implementation (matches nanochat requirement)
   - ✅ `TransformerDecoderLayer` - Decoder-only transformer building blocks
   - ✅ Activation functions: `ReLU`, `GELU` (ReLU² can be implemented via functional API)
   - ✅ `Dropout` - Regularization support

2. **Optimizers** (`aprender::nn::optim`):
   - ✅ `AdamW` - AdamW optimizer with configurable hyperparameters
   - ✅ `SGD` - Stochastic gradient descent
   - ✅ `RMSprop` - RMSprop optimizer

3. **Learning Rate Scheduling** (`aprender::nn::scheduler`):
   - ✅ `WarmupCosineScheduler` - Cosine annealing with warmup
   - ✅ `LinearWarmup` - Linear warmup support
   - ✅ `ReduceLROnPlateau` - Plateau-based reduction

4. **Automatic Differentiation** (`aprender::autograd`):
   - ✅ `Tensor` - Automatic differentiation support
   - ✅ Gradient computation for training
   - ✅ Backward pass implementation

5. **GPU Support**:
   - ✅ `gpu` feature enables wgpu backend via trueno
   - ✅ CPU fallback with SIMD acceleration via trueno
   - ✅ Automatic device selection

**Gaps Identified**:
- ReLU² activation: Not directly available, but can be implemented using `aprender::nn::functional` API
- QK normalization: Not available, must implement using aprender primitives
- Untied embedding/head weights: Standard practice, aprender supports this
- KV cache: Must implement manually using aprender tensors

**Alternatives Considered**:
- Burn: More mature but heavier, less aligned with minimalism principle
- Candle: Good performance but less pure Rust (some C++ dependencies)
- Custom implementation: Too much work, aprender provides 90% of needed functionality

## 2. Trueno & Nalgebra Integration

### Decision: Use trueno and nalgebra as fallback/complement to aprender

**Rationale**: Aprender uses trueno internally for tensor operations, but direct usage may be needed for custom operations. Nalgebra provides linear algebra operations for specialized computations.

**Trueno Capabilities**:

1. **SIMD Acceleration**:
   - ✅ x86: SSE2, AVX, AVX2, AVX-512
   - ✅ ARM: NEON
   - ✅ WASM: SIMD128
   - ✅ Automatic backend selection

2. **GPU Support**:
   - ✅ wgpu backend (Vulkan/Metal/DX12/WebGPU)
   - ✅ Matrix multiplication acceleration (2-10x speedup for large matrices)
   - ⚠️ Element-wise operations: GPU overhead makes CPU faster (use CPU for these)

3. **Tensor Operations**:
   - ✅ Vector operations (add, mul, dot, sum, max)
   - ✅ Matrix operations (matmul, transpose, convolve2d)
   - ✅ Broadcasting support

**Nalgebra Capabilities**:

1. **Linear Algebra**:
   - ✅ Matrix operations (multiplication, decomposition)
   - ✅ Eigenvalue decomposition (for PCA if needed)
   - ✅ Sparse matrix support

2. **Usage Strategy**:
   - Primary: Use through aprender (aprender uses nalgebra internally)
   - Direct: Only if aprender doesn't expose needed operations
   - Avoid: Duplicating functionality already in aprender

## 3. Model Architecture Design

### Decision: Implement GPT architecture using aprender primitives

**Architecture Components**:

1. **Token Embedding**:
   - Use `aprender::nn::Linear` for embedding layer
   - Apply RMSNorm after embedding (aprender::nn::RMSNorm)
   - Untied weights: Separate embedding and LM head (standard aprender pattern)

2. **Transformer Blocks**:
   - Use `aprender::nn::TransformerDecoderLayer` as base
   - Customize for nanochat requirements:
     - Replace LayerNorm with RMSNorm
     - Add QK normalization (custom implementation)
     - Use GroupedQueryAttention instead of MultiHeadAttention
     - Use RotaryPositionEmbedding for positional encoding
     - Use ReLU² activation in MLP (custom via functional API)

3. **Attention Mechanism**:
   - Use `aprender::nn::GroupedQueryAttention` for GQA
   - Implement KV cache using aprender tensors
   - Causal masking via aprender's attention mask support

4. **MLP Layers**:
   - Use `aprender::nn::Linear` layers
   - Implement ReLU² activation: `relu(x) * relu(x)` via functional API
   - Bias-free: Use `Linear` with bias disabled

5. **Output Head**:
   - Use `aprender::nn::Linear` for language model head
   - Untied from embedding (separate parameters)

**Checkpoint Format**:
- Use aprender's `.apr` format for model persistence
- Supports encryption, compression, and signatures
- Memory-mapped loading for fast startup
- Compatible with SafeTensors export

## 4. Numerical Stability & Performance

### Decision: Implement numerical stability checks and performance optimizations

**Numerical Stability Strategies**:

1. **Attention Stability**:
   - Use max-subtraction before softmax (standard practice)
   - Scale by sqrt(d_k) for attention scores
   - QK normalization to prevent extreme values

2. **Softmax Stability**:
   - Subtract max per row before exp (prevents overflow)
   - Clamp logits before softmax if needed

3. **Gradient Stability**:
   - Gradient clipping (implement via aprender optimizer hooks)
   - Learning rate warmup (aprender scheduler support)

4. **Floating Point**:
   - Use f32 for training (standard, aprender default)
   - Consider f16 quantization for inference (aprender supports via `format-quantize` feature)

**Performance Optimization Strategy**:

1. **CPU Path**:
   - Rely on trueno SIMD acceleration (automatic)
   - Use rayon for parallel data loading (aprender `parallel` feature)

2. **GPU Path**:
   - Enable `gpu` feature for matrix operations
   - Use CPU for element-wise operations (GPU overhead too high)
   - Batch operations to amortize GPU transfer costs

3. **Memory Optimization**:
   - KV cache reuse across inference steps
   - Gradient checkpointing for large models (if needed)
   - Streaming data loading for large datasets

**Benchmarking Strategy**:
- Use `criterion` for micro-benchmarks
- Measure end-to-end training throughput
- Compare against reference Python implementation
- Track memory usage and numerical stability

## 5. Testing & Quality Assurance

### Decision: Comprehensive testing strategy with 100% coverage requirement

**Testing Approach**:

1. **Unit Tests**:
   - Test each module independently
   - Mock dependencies where appropriate
   - Test edge cases and error conditions
   - Use `cargo test` with coverage reporting

2. **Integration Tests**:
   - Test complete training pipeline
   - Test inference workflows
   - Test checkpoint save/load
   - Test distributed training (if implemented)

3. **Property-Based Testing**:
   - Use `proptest` for numerical operations
   - Test mathematical properties (e.g., attention is linear in V)
   - Test invariance properties (e.g., RoPE rotation)

4. **Numerical Parity Testing**:
   - Compare outputs with reference Python implementation
   - Tolerance: 1e-4 for logits, 1e-6 for final outputs
   - Test on identical inputs with same random seeds

5. **Performance Testing**:
   - Benchmark critical paths (attention, matmul, forward pass)
   - Track regression over time
   - Compare CPU vs GPU performance

6. **Contract Testing**:
   - Test API contracts (REST endpoints)
   - Test CLI interface contracts
   - Verify streaming behavior

**Coverage Tools**:
- `cargo-tarpaulin` or `cargo-llvm-cov` for coverage reporting
- Enforce 100% coverage for public APIs
- Allow lower coverage for internal implementation details (but still test critical paths)

## 6. Software Development Best Practices

### Decision: Follow Rust best practices and LLMOps workflow

**Code Organization**:
- Modular crate structure (one crate per ML cycle step)
- Clear separation of concerns
- Single responsibility principle per module

**Error Handling**:
- Use `Result<T, E>` for fallible operations
- Use `thiserror` or `anyhow` for error types
- Provide clear error messages

**Documentation**:
- All public APIs must have doc comments
- Include examples in documentation
- Document safety invariants for any unsafe code

**CI/CD**:
- Run tests on every commit
- Enforce formatting (rustfmt)
- Enforce linting (clippy)
- Run benchmarks on performance-critical changes

**Version Control**:
- Semantic versioning
- Clear commit messages
- Feature branches for development
- PR reviews before merge

## 7. LLMOps Workflow Alignment

### Decision: Organize development phases following LLMOps best practices

**Phase Mapping**:

1. **Phase 0: Research & Architecture** (This phase)
   - ✅ Technology selection
   - ✅ Architecture design
   - ✅ Dependency research

2. **Phase 1: Data Preparation & Preprocessing** (Next)
   - Tokenizer implementation
   - Data loading pipeline
   - Data validation

3. **Phase 2: Model Architecture Development**
   - Core GPT implementation
   - Attention mechanisms
   - Normalization layers

4. **Phase 3: Training Pipeline Development**
   - Pretraining implementation
   - Mid-training implementation
   - SFT implementation
   - Optimizer integration

5. **Phase 4: Evaluation & Validation**
   - Benchmark implementations
   - Evaluation metrics
   - Report generation

6. **Phase 5: Deployment & Serving**
   - Inference server
   - CLI interface
   - Caching implementation

7. **Phase 6: Integration & Testing**
   - End-to-end testing
   - Performance validation
   - Documentation

**Benefits**:
- Each phase delivers independently testable value
- Clear progression through ML lifecycle
- Enables parallel development where possible
- Aligns with industry best practices

## 8. OpenAI API Compatibility

### Decision: Implement OpenAI-compatible API format

**Rationale**: OpenAI's Chat Completions API has become the de facto standard for LLM inference APIs. Compatibility enables:
- Seamless integration with existing OpenAI-compatible tools and libraries
- Drop-in replacement for OpenAI API in applications
- Better ecosystem compatibility and adoption

**OpenAI API Requirements**:

1. **Endpoint Format**:
   - ✅ `/v1/chat/completions` - Main chat completion endpoint
   - ✅ `/v1/models` - List available models endpoint
   - ✅ Standard OpenAI request/response format

2. **Request Format**:
   - ✅ `model` parameter (required) - Model identifier
   - ✅ `messages` array with `role` and `content`
   - ✅ `temperature`, `top_p`, `max_tokens` parameters
   - ✅ `stream` boolean for streaming support
   - ✅ `stop`, `presence_penalty`, `frequency_penalty` (optional)
   - ✅ `n` parameter for multiple completions (optional)
   - ✅ Nanochat extensions: `top_k`, `seed` (optional, non-breaking)

3. **Response Format (Non-Streaming)**:
   - ✅ `id` - Unique completion identifier
   - ✅ `object: "chat.completion"`
   - ✅ `created` - Unix timestamp
   - ✅ `model` - Model identifier
   - ✅ `choices` array with `message`, `finish_reason`, `index`
   - ✅ `usage` object with `prompt_tokens`, `completion_tokens`, `total_tokens`

4. **Response Format (Streaming)**:
   - ✅ `object: "chat.completion.chunk"` for streaming chunks
   - ✅ `delta` object in choices (instead of `message`)
   - ✅ `finish_reason: null` for in-progress chunks
   - ✅ Final chunk with `finish_reason` and empty `delta`
   - ✅ `[DONE]` marker at end of stream

5. **Error Format**:
   - ✅ OpenAI-compatible error structure with `error.message`, `error.type`, `error.code`

**Implementation Strategy**:
- Use OpenAI-compatible request/response structures
- Support both streaming and non-streaming modes
- Maintain backward compatibility with nanochat-specific extensions (top_k, seed)
- Validate all OpenAI-required fields are present
- Ensure SSE streaming format matches OpenAI exactly

**Alternatives Considered**:
- Custom API format: Rejected - reduces ecosystem compatibility
- Partial compatibility: Rejected - full compatibility enables better integration

## Summary

All research questions resolved. Aprender provides comprehensive support for GPT architecture with minimal gaps. Custom implementations needed for:
- ReLU² activation (via functional API)
- QK normalization (using aprender primitives)
- KV cache (using aprender tensors)

OpenAI API compatibility ensures seamless integration with existing tools and libraries.

The architecture is feasible, aligns with constitution principles, follows LLMOps best practices, and provides OpenAI-compatible API. Ready to proceed to Phase 1: Design & Contracts.

