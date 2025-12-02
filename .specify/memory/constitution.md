<!--
Sync Impact Report:
Version: 1.0.0 → 1.1.0 (Added Aprender API Reuse principle)
Principles Added:
  - VII. Aprender API Reuse (Mandatory Library-First Approach)
Principles Modified:
  - None
Sections Added:
  - None
Sections Modified:
  - Technology Stack Requirements (added aprender as primary ML framework)
  - Development Workflow (added aprender API verification step)
Templates Updated:
  - ✅ .specify/templates/plan-template.md (Constitution Check section updated with Principle VII)
  - ⚠ .specify/templates/spec-template.md (No changes needed - technology agnostic)
  - ⚠ .specify/templates/tasks-template.md (No changes needed - technology agnostic)
  - ⚠ .specify/templates/checklist-template.md (No changes needed - dynamically generated)
Critical Issues Addressed:
  - CRITICAL: Prohibits custom RNG implementations (must use aprender's seeded StdRng)
  - CRITICAL: Prohibits custom weight initialization (must use aprender::nn::init functions)
  - CRITICAL: Prohibits custom dropout (must use aprender::nn::Dropout or proper StdRng)
  - Architecture-specific code must follow aprender patterns and examples
-->

# Nanochat-Rust Constitution

## Core Principles

### I. Pure Rust (Zero C/C++ Dependencies)

**MUST**: All code MUST be written in pure Rust with zero C/C++ dependencies. This includes:
- No FFI bindings to C/C++ libraries
- No build-time C/C++ compilation steps
- All dependencies MUST be pure Rust crates or Rust-native implementations
- Tensor operations, neural network layers, and mathematical computations MUST use pure Rust implementations or Rust-native frameworks (e.g., Burn, Candle with pure Rust backends, or custom implementations)

**Rationale**: Ensures maximum portability, safety guarantees from Rust's type system and borrow checker, eliminates cross-language boundary overhead, and maintains consistency with Rust ecosystem best practices. This principle aligns with the goal of creating a fully Rust-native machine learning implementation.

### II. Performance & Safety First

**MUST**: All implementations MUST prioritize both performance and safety:
- Leverage Rust's zero-cost abstractions and compile-time guarantees
- Use unsafe code only when absolutely necessary and with comprehensive documentation
- All unsafe blocks MUST include safety invariants documentation
- Performance-critical paths MUST be benchmarked and optimized
- Memory safety violations are non-negotiable: use Rust's ownership system, avoid data races, prevent buffer overflows
- Numerical stability MUST be verified in all mathematical operations

**Rationale**: Rust's unique combination of performance and safety enables building high-performance ML systems without sacrificing correctness. This principle ensures the codebase maintains production-grade quality while achieving competitive performance with C/C++ implementations.

### III. Minimalism & Simplicity

**MUST**: The codebase MUST adhere to minimalism and simplicity principles:
- No giant configuration objects or model factories
- Avoid if-then-else monsters and excessive abstraction layers
- Single, cohesive, readable, hackable codebase structure
- Each module MUST have a clear, single purpose
- Prefer explicit code over clever abstractions
- YAGNI (You Aren't Gonna Need It) principle: don't add features until they're needed
- Code MUST be maximally forkable and understandable by contributors

**Rationale**: Aligns with Karpathy's nanochat philosophy of creating a minimal, hackable baseline rather than an exhaustively configurable framework. Simplicity reduces cognitive load, enables faster iteration, and makes the codebase accessible to a broader audience.

### IV. Classical ML Algorithms

**MUST**: When appropriate, leverage classical machine learning algorithms optimized for Rust:
- Prefer well-understood, proven algorithms over experimental approaches when performance and correctness are equivalent
- Optimize classical algorithms (e.g., optimization methods, numerical linear algebra) for Rust's performance characteristics
- Document algorithm choices with rationale and references
- Ensure all ML algorithms are implemented with numerical stability considerations

**Rationale**: Classical ML algorithms provide a solid foundation with well-understood properties, performance characteristics, and correctness guarantees. Optimizing these for Rust enables building reliable, performant ML systems while maintaining simplicity.

### V. Code Quality & Readability

**MUST**: All code MUST meet high quality and readability standards:
- Comprehensive documentation for public APIs and complex algorithms
- Clear, descriptive naming conventions following Rust idioms
- Consistent code formatting (enforced via rustfmt)
- All public items MUST have doc comments
- Complex logic MUST include inline comments explaining the "why"
- Code reviews MUST verify readability and maintainability

**Rationale**: Readable code is maintainable code. This principle ensures the codebase remains accessible to contributors and can evolve sustainably. Documentation and clear naming reduce the learning curve for new contributors.

### VI. LLM Disclosure Policy

**MUST**: When submitting contributions, declare any parts that had substantial LLM assistance:
- PRs MUST disclose any code sections substantially contributed by LLMs
- Contributors MUST declare if they do not fully understand LLM-generated code
- LLM-assisted code MUST still meet all other constitution principles
- Code quality standards apply equally to human and LLM-generated code

**Rationale**: Transparency in development process ensures maintainability and allows reviewers to provide appropriate scrutiny. This policy aligns with nanochat's disclosure requirements and maintains code quality regardless of generation method.

### VII. Aprender API Reuse (Mandatory Library-First Approach)

**MUST**: All implementations MUST prioritize aprender's built-in APIs before implementing custom solutions:
- **CRITICAL**: Before implementing any ML functionality, MUST verify if aprender provides equivalent functionality
- **CRITICAL**: Weight initialization MUST use `aprender::nn::init::{normal, uniform, xavier_*, kaiming_*}` with proper seeding (`seed: Option<u64>`)
- **CRITICAL**: Dropout MUST use `aprender::nn::Dropout` or proper `rand::rngs::StdRng` with `SeedableRng::seed_from_u64()` - hash-based "RNG" implementations are FORBIDDEN
- **CRITICAL**: Random number generation MUST use `rand::rngs::StdRng` with proper seeding - custom hash-based or LCG implementations are FORBIDDEN
- Normalization layers MUST use aprender's implementations (`aprender::nn::{RMSNorm, LayerNorm, BatchNorm1d, ...}`)
- Optimizers MUST use `aprender::nn::optim::{Adam, AdamW, SGD, ...}`
- Learning rate schedulers MUST use `aprender::nn::scheduler::{WarmupCosineScheduler, LinearWarmup, ...}`
- Loss functions MUST use `aprender::nn::loss::{CrossEntropyLoss, MSELoss, ...}`
- Model serialization MUST use `aprender::nn::serialize::{save_model, load_model}`

**MUST NOT**: The following are FORBIDDEN:
- Custom weight initialization functions that duplicate aprender's `init` module functionality
- Custom dropout implementations using hash-based RNG, `DefaultHasher`, or LCG
- Custom random number generators for ML purposes (use `StdRng` from `rand` crate)
- Reinventing functionality that aprender already provides with proper seeding and thread safety

**Architecture-Specific Code**: When architecture-specific functionality is required (e.g., KV cache, custom RoPE slicing, QK normalization):
- MUST follow aprender's patterns and coding style
- MUST reference aprender examples for similar functionality
- MUST use aprender's underlying primitives (e.g., `Tensor`, `Module` trait) correctly
- MUST document why aprender's standard components cannot be used directly
- MUST optimize implementation following aprender's performance patterns
- SHOULD leverage aprender's helper functions and utilities where applicable

**Verification Process**:
- All PRs MUST include verification that aprender APIs were checked before implementation
- Code reviews MUST flag any custom implementations that duplicate aprender functionality
- Before implementing any ML component, developers MUST:
  1. Search aprender's `src/nn/` module for equivalent functionality
  2. Check aprender examples for usage patterns
  3. Document why aprender's implementation cannot be used (if applicable)
  4. Ensure custom code follows aprender's patterns and uses aprender primitives

**Rationale**: Aprender provides battle-tested, thread-safe, properly seeded implementations of common ML operations. Reinventing this functionality introduces bugs (non-reproducible results, poor statistical properties, thread-safety issues), increases maintenance burden, and misaligns with the project's goal of leveraging Rust-native ML frameworks. Architecture-specific code is acceptable when necessary, but must follow aprender's established patterns to maintain consistency and quality.

## Technology Stack Requirements

**MUST**: The project MUST use:
- Rust as the sole implementation language
- Pure Rust dependencies only (no C/C++ FFI)
- **Aprender as the primary ML framework** for all neural network operations, optimizers, schedulers, loss functions, and model serialization
- Rust-native ML frameworks (aprender is the primary; other pure Rust frameworks may be used only if aprender lacks required functionality)
- Standard Rust tooling: Cargo, rustfmt, clippy, rustc
- Semantic versioning for releases
- `rand` crate's `StdRng` with `SeedableRng` for all random number generation in ML contexts

**MUST NOT**: The project MUST NOT use:
- C/C++ dependencies or FFI bindings
- Python runtime dependencies (Python tooling for data processing is acceptable)
- Proprietary or vendor-locked frameworks that require C/C++ backends
- Build systems that compile C/C++ code
- Custom hash-based RNG implementations (`DefaultHasher`, LCG) for ML purposes
- Custom weight initialization that duplicates aprender's `init` module
- Custom dropout implementations that duplicate aprender's `dropout` module

## Development Workflow

**MUST**: All development MUST follow:
- Code must compile without warnings (enforced via `#![deny(warnings)]` or clippy)
- All public APIs MUST have unit tests
- Integration tests for end-to-end functionality
- Benchmarks for performance-critical paths
- PRs MUST pass all tests and linting before merge
- Code reviews MUST verify constitution compliance
- Breaking changes MUST be documented and versioned appropriately
- **Aprender API verification**: Before implementing any ML functionality, developers MUST verify aprender's API availability and use it when available
- **No wheel reinvention**: PRs MUST be rejected if they implement functionality that aprender already provides (see Principle VII)

**SHOULD**: Development SHOULD:
- Use feature flags for experimental functionality
- Maintain backward compatibility when possible
- Provide migration guides for breaking changes
- Include performance benchmarks in PRs affecting critical paths

## Governance

This constitution supersedes all other development practices and guidelines. All code, documentation, and processes MUST comply with these principles.

**Amendment Procedure**:
- Amendments require documentation of rationale and impact assessment
- Version MUST be incremented per semantic versioning (MAJOR.MINOR.PATCH)
- All amendments MUST be reviewed and approved before adoption
- Breaking changes to principles require MAJOR version bump

**Compliance Review**:
- All PRs MUST be reviewed for constitution compliance
- Constitution violations are blocking issues and MUST be resolved before merge
- Regular audits SHOULD be conducted to ensure ongoing compliance

**Version**: 1.1.0 | **Ratified**: 2025-01-27 | **Last Amended**: 2025-01-27
