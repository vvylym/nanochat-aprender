# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Rust [version from rustc --version or NEEDS CLARIFICATION]  
**Primary Dependencies**: [Pure Rust crates only, e.g., Burn, serde, tokio or NEEDS CLARIFICATION - MUST be pure Rust]  
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]  
**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]  
**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]
**Project Type**: [single/web/mobile - determines source structure]  
**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]  
**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Pure Rust (Zero C/C++ Dependencies)**
- [ ] All dependencies are pure Rust crates (no FFI, no C/C++ compilation)
- [ ] No build-time C/C++ steps required
- [ ] All ML operations use Rust-native implementations

**II. Performance & Safety First**
- [ ] Unsafe code usage documented with safety invariants (if any)
- [ ] Performance-critical paths identified and benchmarked
- [ ] Memory safety verified (ownership, no data races, no buffer overflows)

**III. Minimalism & Simplicity**
- [ ] No excessive abstraction layers or configuration objects
- [ ] Each module has clear, single purpose
- [ ] Code is explicit and readable (no clever abstractions)
- [ ] YAGNI principle applied (no premature features)

**IV. Classical ML Algorithms**
- [ ] Algorithm choices documented with rationale
- [ ] Numerical stability considerations addressed
- [ ] Performance optimizations for Rust characteristics

**V. Code Quality & Readability**
- [ ] Public APIs have comprehensive documentation
- [ ] Naming follows Rust idioms
- [ ] Code formatting enforced (rustfmt)
- [ ] Complex logic includes inline comments

**VI. LLM Disclosure Policy**
- [ ] Any LLM-assisted code sections declared in PR

**VII. Aprender API Reuse (Mandatory Library-First Approach)**
- [ ] Aprender APIs checked before implementing any ML functionality
- [ ] Weight initialization uses `aprender::nn::init` functions with proper seeding
- [ ] Dropout uses `aprender::nn::Dropout` or proper `StdRng` (no hash-based RNG)
- [ ] Random number generation uses `rand::rngs::StdRng` with `SeedableRng` (no custom RNG)
- [ ] Normalization, optimizers, schedulers, loss functions use aprender implementations
- [ ] Architecture-specific code follows aprender patterns and examples
- [ ] Custom implementations documented with rationale for why aprender cannot be used

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
