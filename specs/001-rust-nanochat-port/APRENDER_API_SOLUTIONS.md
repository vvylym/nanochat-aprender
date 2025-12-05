# Aprender API Solutions for Known Limitations

**Date**: 2024-12-19  
**Status**: Proposed Solutions  
**Priority**: HIGH

## Overview

This document provides concrete solutions for two known limitations in the Phase 4 pretraining implementation, based on analysis of the `aprender` framework API.

## Limitation 1: Gradient Clipping (P4)

### Current Status
- **Placeholder**: `clip_gradients()` function returns `0.0` and does not perform actual clipping
- **Location**: `crates/nanochat-pretrain/src/train.rs:200-210`
- **Impact**: Training may be unstable without gradient clipping

### Aprender API Analysis

**Available APIs:**
1. `aprender::autograd::get_grad(id: TensorId) -> Option<Tensor>` - Get gradient for a tensor by ID
2. `tensor.grad() -> Option<&Tensor>` - Get gradient tensor directly
3. `tensor.data() -> &[f32]` - Access tensor data
4. `tensor.id() -> TensorId` - Get tensor ID for gradient lookup

**Key Finding**: Aprender **does** provide gradient access via `get_grad()` and `tensor.grad()`. The limitation was due to incomplete API exploration, not missing functionality.

### Proposed Solution

**File**: `crates/nanochat-pretrain/src/train.rs`

**Current API Limitation**: `get_grad()` returns a cloned `Tensor`, so modifying it doesn't affect the computation graph. We need to check if aprender provides a way to set gradients back.

**Practical Implementation (Compute Norm Only - For Now):**

```rust
use aprender::autograd::get_grad;
use aprender::nn::Module;

/// Clip gradients by global norm (matching PyTorch's `torch.nn.utils.clip_grad_norm_`)
///
/// **Current Implementation**: Computes and returns the global gradient norm.
/// Actual clipping requires aprender API support for setting gradients.
///
/// # Arguments
/// * `model` - The GPT model
/// * `max_norm` - Maximum gradient norm (clipping threshold)
///
/// # Returns
/// The global gradient norm (for logging/monitoring)
///
/// # Algorithm
/// 1. Compute global norm: sqrt(sum(grad²) for all parameters)
/// 2. If norm > max_norm: Log warning (clipping not yet implemented)
///
/// # Python Reference
/// Matches PyTorch's `torch.nn.utils.clip_grad_norm_(parameters, max_norm)`
fn clip_gradients(model: &GPT, max_norm: f32) -> Result<f32> {
    if max_norm <= 0.0 {
        return Ok(0.0);
    }

    // Get all model parameters
    let parameters = model.parameters();
    
    // Compute global norm
    let mut total_norm_sq = 0.0;
    
    for param in parameters {
        if let Some(grad) = get_grad(param.id()) {
            let grad_data = grad.data();
            // Sum of squares for this parameter's gradient
            let param_norm_sq: f32 = grad_data.iter().map(|&x| x * x).sum();
            total_norm_sq += param_norm_sq;
        }
    }
    
    let total_norm = total_norm_sq.sqrt();
    
    // Log warning if clipping would be needed
    if total_norm > max_norm {
        eprintln!(
            "Warning: Gradient norm ({:.4}) exceeds max_norm ({:.4}). \
             Clipping not yet implemented - requires aprender API for setting gradients.",
            total_norm, max_norm
        );
    }
    
    Ok(total_norm)
}
```

**Required Aprender API Addition:**

To fully implement gradient clipping, aprender needs one of:

1. **Option A**: Public method to set gradients:
   ```rust
   pub fn set_grad(&mut self, grad: Tensor) {
       self.grad = Some(Box::new(grad));
   }
   ```

2. **Option B**: Public method to scale gradients in-place:
   ```rust
   pub fn scale_grad(&mut self, scale: f32) {
       if let Some(ref mut grad) = self.grad {
           // Scale gradient in-place
           for val in grad.data_mut() {
               *val *= scale;
           }
       }
   }
   ```

3. **Option C**: Graph-level method to update gradients:
   ```rust
   pub fn update_grad(&mut self, id: TensorId, grad: Tensor) {
       if let Some(tensor) = self.tensors.get_mut(&id) {
           tensor.set_grad(grad);
       }
   }
   ```

**Important Note on Gradient Modification:**

The `get_grad()` function returns an `Option<Tensor>`, which is an owned copy. Modifying this tensor **will not** affect the gradients stored in the computation graph. We need to modify gradients in the graph itself.

**Alternative Approach (Recommended - Modify via Graph):**

Since `get_grad()` returns a copy, we need to:
1. Compute the global norm from all gradients
2. If clipping needed, scale each gradient and update it in the graph
3. Use `clear_grad()` and then manually set scaled gradients (if aprender supports this)

**Practical Implementation (Using Parameter Gradients Directly):**

A simpler approach is to iterate through parameters and access their gradients directly:

```rust
use aprender::autograd::get_grad;
use aprender::nn::Module;

fn clip_gradients(model: &GPT, max_norm: f32) -> Result<f32> {
    if max_norm <= 0.0 {
        return Ok(0.0);
    }

    let parameters = model.parameters();
    
    // Step 1: Compute global norm
    let mut total_norm_sq = 0.0;
    let mut param_grads = Vec::new();
    
    for param in parameters {
        if let Some(grad) = get_grad(param.id()) {
            let grad_data = grad.data();
            let param_norm_sq: f32 = grad_data.iter().map(|&x| x * x).sum();
            total_norm_sq += param_norm_sq;
            param_grads.push((param.id(), grad));
        }
    }
    
    let total_norm = total_norm_sq.sqrt();
    
    // Step 2: Clip if needed
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        
        // Step 3: Scale gradients
        // Since get_grad() returns a copy, we need to:
        // 1. Clear existing gradients
        // 2. Create scaled gradients
        // 3. Set them back (if aprender supports this)
        
        // For now, we'll need to check aprender's API for setting gradients
        // This might require using aprender's operations to create scaled gradients
        // and then somehow updating the graph
        
        // TODO: Verify if aprender has a way to set gradients after clearing
        // If not, we may need to modify the optimizer to apply clipping
    }
    
    Ok(total_norm)
}
```

**Workaround: Apply Clipping in Optimizer Wrapper**

If direct gradient modification isn't possible, create an optimizer wrapper that applies clipping:

```rust
pub struct ClippedAdamW {
    inner: AdamW,
    max_norm: f32,
}

impl ClippedAdamW {
    pub fn new(params: Vec<&mut Tensor>, lr: f32, max_norm: f32) -> Self {
        Self {
            inner: AdamW::new(params, lr),
            max_norm,
        }
    }
    
    pub fn step(&mut self, model: &GPT) -> Result<f32> {
        // Compute norm and clip before optimizer step
        let norm = compute_gradient_norm(model)?;
        if norm > self.max_norm {
            // Scale gradients (implementation depends on aprender API)
            scale_gradients(model, self.max_norm / norm)?;
        }
        self.inner.step();
        Ok(norm)
    }
}
```

**Testing Requirements:**
- Unit test: Verify norm computation is correct
- Unit test: Verify clipping occurs when norm > max_norm
- Unit test: Verify no clipping when norm < max_norm
- Integration test: Verify training stability improves with clipping

---

## Limitation 2: Optimizer State Serialization (P6)

### Current Status
- **Partial**: Only step count and learning rate are saved
- **Missing**: Moment estimates (m, v) for AdamW optimizer
- **Location**: `crates/nanochat-pretrain/src/train.rs:359-366`
- **Impact**: On resume, optimizer restarts with fresh moment estimates, causing brief adjustment period

### Aprender API Analysis

**Current AdamW Structure:**
```rust
pub struct AdamW {
    param_ids: Vec<TensorId>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,      // First moment estimates (private)
    v: Vec<Vec<f32>>,      // Second moment estimates (private)
    t: usize,              // Step counter (private)
    initialized: bool,     // Initialization flag (private)
}
```

**Key Finding**: The `m` and `v` fields are **private** and there are no public getter/setter methods. The `t` field is also private.

### Proposed Solutions

#### Solution A: Add Public Accessors to Aprender (Recommended)

**File**: `.idea/aprender/src/nn/optim.rs` (requires aprender modification)

```rust
impl AdamW {
    // ... existing methods ...
    
    /// Get optimizer state for serialization
    ///
    /// Returns a serializable representation of the optimizer's internal state.
    pub fn get_state(&self) -> AdamWState {
        AdamWState {
            step: self.t,
            lr: self.lr,
            m: self.m.clone(),
            v: self.v.clone(),
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
    
    /// Restore optimizer state from serialized representation
    ///
    /// # Arguments
    /// * `state` - Serialized optimizer state
    /// * `params` - Current model parameters (must match original parameter order)
    ///
    /// # Errors
    /// Returns error if parameter count doesn't match state
    pub fn restore_state(&mut self, state: AdamWState, params: &[&Tensor]) -> Result<()> {
        if state.m.len() != params.len() {
            anyhow::bail!(
                "Parameter count mismatch: state has {}, model has {}",
                state.m.len(),
                params.len()
            );
        }
        
        // Verify param_ids match (optional, for safety)
        let current_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        if current_ids != self.param_ids {
            // Log warning but continue (parameter order might have changed)
            eprintln!("Warning: Parameter IDs don't match, state may be invalid");
        }
        
        self.t = state.step;
        self.lr = state.lr;
        self.m = state.m;
        self.v = state.v;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.eps = state.eps;
        self.weight_decay = state.weight_decay;
        self.initialized = true;
        
        Ok(())
    }
}

/// Serializable optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWState {
    pub step: usize,
    pub lr: f32,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}
```

**Pros:**
- Clean API
- Type-safe
- Maintains aprender's encapsulation

**Cons:**
- Requires modifying aprender (external dependency)
- May require upstream PR/contribution

#### Solution B: Use Reflection/Serialization Trait (If Available)

If aprender provides a serialization trait for optimizers, we could use that:

```rust
// Hypothetical - check if aprender has this
impl Serialize for AdamW {
    // ...
}
```

**Status**: Need to check if aprender has such a trait.

#### Solution C: Workaround - Save State via Optimizer Wrapper (Current Workaround)

**File**: `crates/nanochat-pretrain/src/optimizer.rs`

Create a wrapper that tracks state separately:

```rust
use serde::{Serialize, Deserialize};

/// Wrapper around AdamW that tracks state for serialization
pub struct SerializableAdamW {
    optimizer: AdamW,
    // Track state separately for serialization
    state: AdamWStateSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdamWStateSnapshot {
    step: usize,
    lr: f32,
    // Note: m and v still not accessible without aprender API
}

impl SerializableAdamW {
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let optimizer = AdamW::new(params, lr);
        Self {
            optimizer,
            state: AdamWStateSnapshot {
                step: 0,
                lr,
            },
        }
    }
    
    pub fn step(&mut self) {
        self.optimizer.step();
        self.state.step += 1;
    }
    
    pub fn get_state(&self) -> &AdamWStateSnapshot {
        &self.state
    }
    
    // Delegate other methods to optimizer
    pub fn zero_grad(&mut self) {
        self.optimizer.zero_grad();
    }
    
    pub fn lr(&self) -> f32 {
        self.optimizer.lr()
    }
    
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
        self.state.lr = lr;
    }
}
```

**Pros:**
- Works with current aprender API
- No aprender modification needed

**Cons:**
- Still can't save/restore m and v
- Adds wrapper complexity
- Doesn't solve the core problem

#### Solution D: Fork/Extend Aprender (Not Recommended)

Fork aprender and add the necessary methods. This is not recommended as it:
- Creates maintenance burden
- Diverges from upstream
- May cause compatibility issues

### Recommended Approach

**Short-term (Current Implementation):**
- Continue with current workaround (save step and LR only)
- Document the limitation clearly
- Accept brief adjustment period on resume

**Medium-term (Preferred):**
- **Propose PR to aprender**: Add `get_state()` and `restore_state()` methods to `AdamW`
- This is a reasonable feature request that benefits the entire aprender ecosystem
- Follow aprender's contribution guidelines

**Long-term:**
- If aprender PR is accepted, update implementation to use new API
- Remove workaround code
- Add comprehensive tests for state serialization

---

## Implementation Priority

### P4 (Gradient Clipping) - **HIGH PRIORITY**

**Action Items:**
1. ✅ Verify `get_grad()` API works as expected
2. ⏳ Implement `clip_gradients()` using `get_grad()`
3. ⏳ Add unit tests for gradient clipping
4. ⏳ Add integration test for training stability
5. ⏳ Update documentation

**Estimated Effort**: 2-3 hours

### P6 (Optimizer State) - **MEDIUM PRIORITY**

**Action Items:**
1. ⏳ Check if aprender has serialization traits
2. ⏳ Draft PR proposal for aprender (add `get_state()`/`restore_state()`)
3. ⏳ If PR accepted, implement state serialization
4. ⏳ If PR rejected, document limitation and accept workaround

**Estimated Effort**: 
- PR proposal: 1-2 hours
- Implementation (if PR accepted): 2-3 hours
- Total: 3-5 hours

---

## Testing Strategy

### Gradient Clipping Tests

```rust
#[test]
fn test_clip_gradients_norm_below_threshold() {
    // Create model with gradients
    // Verify no clipping occurs when norm < max_norm
}

#[test]
fn test_clip_gradients_norm_above_threshold() {
    // Create model with large gradients
    // Verify gradients are scaled correctly
}

#[test]
fn test_clip_gradients_computes_correct_norm() {
    // Verify global norm computation is correct
}
```

### Optimizer State Tests

```rust
#[test]
fn test_optimizer_state_serialization() {
    // Save optimizer state
    // Restore optimizer state
    // Verify m, v, step, lr match
}

#[test]
fn test_optimizer_state_parameter_mismatch() {
    // Verify error handling for parameter count mismatch
}
```

---

## References

- **Aprender Source**: `.idea/aprender/src/nn/optim.rs`
- **Aprender Autograd**: `.idea/aprender/src/autograd/mod.rs`
- **Current Implementation**: `crates/nanochat-pretrain/src/train.rs`
- **Remediation Plan**: `specs/001-rust-nanochat-port/PHASE4_REMEDIATION_PLAN.md`

---

## Next Steps

1. **Immediate**: Implement gradient clipping using `get_grad()` API
2. **Short-term**: Test gradient clipping in training loop
3. **Medium-term**: Propose aprender PR for optimizer state serialization
4. **Long-term**: Update implementation once aprender PR is merged

