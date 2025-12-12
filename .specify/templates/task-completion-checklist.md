# Task Completion Checklist

**Task ID**: [TXXX]  
**Task Description**: [Brief description]  
**Phase**: [Phase number/name]  
**Completion Date**: [YYYY-MM-DD]  
**Completed By**: [Name/GitHub username]

## Pre-Completion Verification

### 1. Code Implementation
- [ ] All code is implemented (no `TODO`, `FIXME`, `XXX`, `TKTK`, `placeholder` comments)
- [ ] No hardcoded values (all configurable via config file or CLI args)
- [ ] All functions/methods have proper error handling
- [ ] All public APIs have doc comments
- [ ] Code follows Rust idioms and best practices

### 2. Reference Implementation Alignment
- [ ] Compared against Python reference implementation (if applicable)
- [ ] All equivalent functionality is implemented
- [ ] Behavior matches reference (or differences are documented and justified)
- [ ] No missing critical features from reference

### 3. Testing
- [ ] Unit tests exist and pass (`cargo test`)
- [ ] Integration tests exist and pass (if applicable)
- [ ] Edge cases are covered
- [ ] Error cases are tested
- [ ] All tests use `.expect()` instead of `.unwrap()` (per constitution)

### 4. Quality Gates
- [ ] Code formatting: `cargo fmt --all` passes
- [ ] Linting: `cargo clippy --workspace --all-features --all-targets` passes (no warnings)
- [ ] Tests: `cargo test --workspace --all-features` passes
- [ ] No compiler warnings

### 5. Documentation
- [ ] Public APIs have comprehensive doc comments
- [ ] Complex logic has inline comments explaining "why"
- [ ] README/examples updated if user-facing changes
- [ ] Architecture decisions documented (if significant)

### 6. Configuration & Dependencies
- [ ] All hyperparameters/config values are loadable from config file
- [ ] No magic numbers (constants are named and documented)
- [ ] Dependencies are properly declared in `Cargo.toml`
- [ ] No unused dependencies

### 7. Error Handling
- [ ] All `Result` types are properly handled
- [ ] Error messages are descriptive and actionable
- [ ] No `.unwrap()` calls (use `.expect()` with descriptive messages)
- [ ] Error propagation uses `?` operator appropriately

### 8. Performance & Safety
- [ ] No unsafe code (or unsafe code is documented with safety invariants)
- [ ] Memory safety verified (no leaks, no use-after-free)
- [ ] Numerical stability considered (if applicable)
- [ ] Performance-critical paths are benchmarked (if applicable)

## Post-Completion Verification

### 9. Integration
- [ ] Changes integrate with existing codebase
- [ ] No breaking changes to public APIs (or breaking changes are documented)
- [ ] Dependent tasks can proceed (if this task blocks others)

### 10. Review
- [ ] Code review completed (if applicable)
- [ ] All review comments addressed
- [ ] Constitution compliance verified

## Sign-Off

**Completed By**: _________________  
**Date**: _________________  
**Verified By**: _________________ (if applicable)

## Notes

[Any additional notes, known limitations, or future improvements]

