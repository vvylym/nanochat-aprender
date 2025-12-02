//! Numerical stability checks and validation
//!
//! This module provides utilities for detecting numerical issues such as
//! NaN, Inf, overflow, and underflow in tensor operations.

use aprender::autograd::Tensor;
use anyhow::{Context, Result};

/// Check if a tensor contains any NaN values
///
/// # Arguments
/// * `tensor` - Tensor to check
///
/// # Returns
/// `true` if tensor contains NaN, `false` otherwise
pub fn has_nan(tensor: &Tensor) -> bool {
    tensor.data().iter().any(|&x| x.is_nan())
}

/// Check if a tensor contains any Inf values
///
/// # Arguments
/// * `tensor` - Tensor to check
///
/// # Returns
/// `true` if tensor contains Inf, `false` otherwise
pub fn has_inf(tensor: &Tensor) -> bool {
    tensor.data().iter().any(|&x| x.is_infinite())
}

/// Validate that a tensor doesn't contain NaN or Inf values
///
/// # Arguments
/// * `tensor` - Tensor to validate
/// * `name` - Name of the tensor for error messages
///
/// # Returns
/// Error if tensor contains NaN or Inf
pub fn validate_tensor(tensor: &Tensor, name: &str) -> Result<()> {
    if has_nan(tensor) {
        anyhow::bail!("Tensor '{}' contains NaN values", name);
    }
    if has_inf(tensor) {
        anyhow::bail!("Tensor '{}' contains Inf values", name);
    }
    Ok(())
}

/// Check for potential overflow in tensor values
///
/// # Arguments
/// * `tensor` - Tensor to check
/// * `max_value` - Maximum expected value (default: 1e6)
///
/// # Returns
/// `true` if any value exceeds max_value
pub fn check_overflow(tensor: &Tensor, max_value: f32) -> bool {
    tensor.data().iter().any(|&x| x.abs() > max_value)
}

/// Check for potential underflow in tensor values
///
/// # Arguments
/// * `tensor` - Tensor to check
/// * `min_value` - Minimum expected value (default: 1e-6)
///
/// # Returns
/// `true` if any value is below min_value (and not zero)
pub fn check_underflow(tensor: &Tensor, min_value: f32) -> bool {
    tensor.data().iter().any(|&x| x != 0.0 && x.abs() < min_value)
}

/// Validate tensor values are within expected range
///
/// # Arguments
/// * `tensor` - Tensor to validate
/// * `name` - Name of the tensor for error messages
/// * `min_value` - Minimum allowed value
/// * `max_value` - Maximum allowed value
///
/// # Returns
/// Error if tensor values are outside the range
pub fn validate_range(tensor: &Tensor, name: &str, min_value: f32, max_value: f32) -> Result<()> {
    if check_overflow(tensor, max_value) {
        anyhow::bail!(
            "Tensor '{}' contains values exceeding maximum {}",
            name,
            max_value
        );
    }
    if check_underflow(tensor, min_value) {
        anyhow::bail!(
            "Tensor '{}' contains values below minimum {}",
            name,
            min_value
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tensor() {
        let tensor = Tensor::ones(&[2, 3]);
        // Should not error for normal tensor
        assert!(validate_tensor(&tensor, "test").is_ok());
    }

    #[test]
    fn test_validate_range() {
        let tensor = Tensor::ones(&[2, 3]);
        // Should not error for values in range
        assert!(validate_range(&tensor, "test", 0.0, 10.0).is_ok());
    }
}

