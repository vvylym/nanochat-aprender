# Remediation Plan for Phase 3 Placeholder Issues

## Summary

This document provides concrete remediation steps for 8 critical issues identified in the `nanochat-model` crate where tasks are marked complete but contain placeholder implementations.

## Key Findings from Aprender Codebase

1. **Checkpoint Format**: Aprender uses **SafeTensors** format (`.safetensors`), not `.apr`. The `.apr` format doesn't exist - it's `.apbundle` for bundling multiple models, but for single model checkpoints, SafeTensors is used.

2. **Existing Functionality Available**:
   - `aprender::nn::serialize::state_dict()` - Extract weights from modules
   - `aprender::nn::serialize::load_state_dict_into()` - Load weights into modules
   - `aprender::nn::serialize::save_model()` - Save to SafeTensors
   - `aprender::nn::serialize::load_model()` - Load from SafeTensors
   - `aprender::nn::loss::CrossEntropyLoss` - Cross-entropy loss function
   - `Tensor::data()` - Access tensor data as `&[f32]` for NaN/Inf checks

3. **Missing Functionality**:
   - Tensor concatenation - needs manual implementation
   - GPT doesn't implement `Module` trait - required for `state_dict()`

---

## Issue C1-C2: Checkpoint Weight Management

### Problem
- `extract_weights()` returns empty HashMap
- `load_weights()` is a no-op
- Cannot actually save/load model weights

### Solution
Use aprender's built-in serialization instead of reimplementing:

```rust
// In checkpoint.rs - REPLACE extract_weights and load_weights
use aprender::nn::serialize::{state_dict, load_state_dict_into, StateDict};

fn extract_weights(model: &GPT) -> Result<StateDict> {
    // GPT needs to implement Module trait first
    Ok(state_dict(model, ""))
}

fn load_weights(model: &mut GPT, state: &StateDict) -> Result<()> {
    load_state_dict_into(model, state, "")
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))
}
```

### Required Changes
1. Make GPT implement `Module` trait (see C5)
2. Update `CheckpointData` to use `StateDict` instead of `HashMap<String, Vec<f32>>`
3. Use SafeTensors format instead of JSON (or keep JSON for metadata, SafeTensors for weights)

---

## Issue C3: Placeholder Checkpoint Tests

### Problem
All 6 tests in `test_checkpoint.rs` are placeholders with `assert!(true)`

### Solution
Replace with real tests using actual checkpoint functionality:

```rust
// In test_checkpoint.rs - REPLACE all tests
use nanochat_model::{GPT, GPTConfig, save_checkpoint, load_checkpoint, CheckpointMetadata};
use tempfile::TempDir;

#[test]
fn test_checkpoint_save() {
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model.safetensors");

    let metadata = CheckpointMetadata {
        step: 100,
        loss: Some(2.5),
        learning_rate: Some(0.001),
        extra: HashMap::new(),
    };

    save_checkpoint(&model, &checkpoint_path, Some(metadata.clone())).unwrap();
    assert!(checkpoint_path.exists());
}

#[test]
fn test_checkpoint_load() {
    let config = GPTConfig::default();
    let model = GPT::new(config);
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("model.safetensors");

    save_checkpoint(&model, &checkpoint_path, None).unwrap();
    let (loaded_model, _) = load_checkpoint(&checkpoint_path).unwrap();
    
    assert_eq!(loaded_model.config(), model.config());
}

// ... (implement remaining 4 tests similarly)
```

---

## Issue C4: Numerical Stability Checks

### Problem
All stability check functions return `false` (placeholders)

### Solution
Implement using `Tensor::data()` which returns `&[f32]`:

```rust
// In stability.rs - REPLACE placeholder implementations
pub fn has_nan(tensor: &Tensor) -> bool {
    tensor.data().iter().any(|&x| x.is_nan())
}

pub fn has_inf(tensor: &Tensor) -> bool {
    tensor.data().iter().any(|&x| x.is_infinite())
}

pub fn check_overflow(tensor: &Tensor, max_value: f32) -> bool {
    tensor.data().iter().any(|&x| x.abs() > max_value)
}

pub fn check_underflow(tensor: &Tensor, min_value: f32) -> bool {
    tensor.data().iter().any(|&x| x != 0.0 && x.abs() < min_value)
}
```

---

## Issue C5: Training Mode Forward Pass

### Problem
Forward pass with `targets` returns error instead of computing loss

### Solution
Use aprender's `CrossEntropyLoss`:

```rust
// In gpt.rs - REPLACE loss computation section
use aprender::nn::loss::CrossEntropyLoss;

// In forward() method, replace:
if let Some(targets) = targets {
    // TODO: Implement cross-entropy loss
    anyhow::bail!("Loss computation not yet implemented");
}

// With:
if let Some(targets) = targets {
    // Reshape logits from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
    let batch = logits.shape()[0];
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];
    let logits_flat = logits.view(&[batch * seq_len, vocab_size]);
    
    // Reshape targets from [batch, seq_len] to [batch * seq_len]
    let targets_flat = targets.view(&[batch * seq_len]);
    
    // Compute cross-entropy loss
    let criterion = CrossEntropyLoss::new();
    let loss = criterion.forward(&logits_flat, &targets_flat);
    
    return Ok(loss);
}
```

**Note**: This requires `targets` to be `Tensor` with integer class indices (as f32), matching aprender's API.

---

## Issue C6: KV Cache Concatenation

### Problem
KV cache overwrites instead of concatenating, breaking autoregressive generation

### Solution
Implement manual concatenation along sequence dimension:

```rust
// In attention.rs - REPLACE insert_kv method
pub fn insert_kv(&mut self, layer_idx: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
    while self.cache.len() <= layer_idx {
        self.cache.push((Tensor::zeros(&[0]), Tensor::zeros(&[0])));
    }

    let (cached_k, cached_v) = &self.cache[layer_idx];
    
    if cached_k.shape()[0] == 0 {
        // First insert - just store
        self.cache[layer_idx] = (k.clone(), v.clone());
        return Ok((k, v));
    }

    // Concatenate along sequence dimension (dim 2 for [batch, n_kv_heads, seq_len, head_dim])
    // Manual concatenation: combine data arrays and reshape
    let k_shape = cached_k.shape();
    let v_shape = cached_v.shape();
    let new_k_shape = k.shape();
    let new_v_shape = v.shape();
    
    // Verify shapes are compatible (batch, heads, head_dim must match)
    assert_eq!(k_shape[0], new_k_shape[0], "Batch size mismatch");
    assert_eq!(k_shape[1], new_k_shape[1], "Head count mismatch");
    assert_eq!(k_shape[3], new_k_shape[3], "Head dim mismatch");
    
    // Concatenate data along sequence dimension
    let mut k_data = cached_k.data().to_vec();
    k_data.extend_from_slice(k.data());
    let new_seq_len = k_shape[2] + new_k_shape[2];
    let k_concat = Tensor::new(&k_data, &[k_shape[0], k_shape[1], new_seq_len, k_shape[3]]);
    
    let mut v_data = cached_v.data().to_vec();
    v_data.extend_from_slice(v.data());
    let v_concat = Tensor::new(&v_data, &[v_shape[0], v_shape[1], new_seq_len, v_shape[3]]);
    
    self.cache[layer_idx] = (k_concat.clone(), v_concat.clone());
    Ok((k_concat, v_concat))
}
```

---

## Issue C7: Checkpoint Format Mismatch

### Problem
Task requires `.apr` format but aprender uses SafeTensors

### Solution Options:

**Option A**: Use SafeTensors (recommended - matches aprender)
```rust
// Update checkpoint.rs to use SafeTensors
use aprender::nn::serialize::{save_model, load_model, StateDict};

pub fn save_checkpoint<P: AsRef<Path>>(
    model: &GPT,
    path: P,
    metadata: Option<CheckpointMetadata>,
) -> Result<()> {
    let path = path.as_ref();
    
    // Save weights to SafeTensors
    let weights_path = path.with_extension("safetensors");
    save_model(model, &weights_path)
        .map_err(|e| anyhow::anyhow!("Failed to save weights: {}", e))?;
    
    // Save metadata to JSON
    if let Some(meta) = metadata {
        let meta_path = path.with_extension("json");
        let json = serde_json::to_string_pretty(&meta)?;
        fs::write(&meta_path, json)?;
    }
    
    Ok(())
}
```

**Option B**: Update task description to reflect SafeTensors format

**Recommendation**: Use Option A (SafeTensors) as it's the aprender standard.

---

## Issue C8: RoPE Slicing

### Problem
Uses full precomputed embeddings instead of slicing to sequence length

### Solution Options:

**Option A**: Implement slicing using `view()` and manual data extraction
```rust
// In gpt.rs forward() method
// Get RoPE embeddings for current sequence length
let cos = self.cos.as_ref().unwrap();
let sin = self.sin.as_ref().unwrap();

// Extract slice: cos/sin are [1, max_seq_len, 1, head_dim/2]
// We need [1, seq_len, 1, head_dim/2]
let cos_data = cos.data();
let sin_data = sin.data();
let head_dim_half = cos.shape()[3];
let cos_slice_data: Vec<f32> = cos_data[..seq_len * head_dim_half].to_vec();
let sin_slice_data: Vec<f32> = sin_data[..seq_len * head_dim_half].to_vec();
let cos_slice = Tensor::new(&cos_slice_data, &[1, seq_len, 1, head_dim_half]);
let sin_slice = Tensor::new(&sin_slice_data, &[1, seq_len, 1, head_dim_half]);
```

**Option B**: Document that full embeddings are acceptable for now (less efficient but correct)

**Recommendation**: Option A for correctness, but can defer if performance is acceptable.

---

## Additional Required Changes

### Make GPT Implement Module Trait

This is required for checkpoint functionality:

```rust
// In gpt.rs - ADD Module implementation
use aprender::nn::Module;

impl Module for GPT {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Note: This conflicts with our custom forward() signature
        // We may need to keep forward() as-is and add a separate method
        // Or wrap it appropriately
        todo!("Module::forward needs to be implemented or GPT should not implement Module")
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        // Add embedding weights
        params.push(&self.wte.weight);
        
        // Add block parameters (attention, MLP)
        for block in &self.blocks {
            // Need to expose block parameters
            // This requires Block to also implement Module or expose parameters
        }
        
        // Add LM head parameters
        // Need to expose lm_head.projection parameters
        
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        // Similar to parameters() but mutable
        todo!("Implement parameters_mut")
    }
}
```

**Note**: This is complex because GPT's structure doesn't directly expose all parameters. Consider:
1. Making Block, TokenEmbedding, LanguageModelHead implement Module
2. Collecting parameters from sub-modules
3. Or using a different approach for checkpointing that doesn't require Module

---

## Implementation Priority

1. **C4** (Stability checks) - Easiest, just use `Tensor::data()`
2. **C5** (Training loss) - Use existing `CrossEntropyLoss`
3. **C6** (KV cache) - Manual concatenation
4. **C1-C2** (Checkpoint weights) - Requires Module trait implementation
5. **C3** (Tests) - Depends on C1-C2 being fixed
6. **C7** (Format) - Update to SafeTensors
7. **C8** (RoPE slicing) - Can defer if performance acceptable

---

## Unnecessary Reimplementations to Remove

1. **Custom checkpoint format** - Use aprender's SafeTensors instead
2. **Custom loss computation** - Use `CrossEntropyLoss` instead
3. **Custom weight extraction** - Use `state_dict()` instead
4. **Placeholder stability checks** - Use `Tensor::data()` directly

---

## Next Steps

1. Review this plan
2. Implement fixes in priority order
3. Update task descriptions if format requirements change
4. Re-run tests to verify all placeholders are removed
5. Update tasks.md to mark as complete only after all placeholders removed

