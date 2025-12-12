//! Weight initialization helpers for matching Python nanochat's scheme
//!
//! This module provides initialization schemes that match the Python reference implementation.
//!
//! # Aprender API Compliance (Principle VII)
//!
//! **Note**: This module uses custom initialization because:
//! 1. `aprender::nn::init::normal()` and `aprender::nn::init::uniform()` are `pub(crate)` (not public)
//! 2. The Python reference uses custom formulas that don't match aprender's public functions:
//!    - `xavier_normal`: std = sqrt(2 / (fan_in + fan_out))
//!    - `kaiming_normal`: std = sqrt(2 / fan_in)
//!    - Our formula: std = 1.0 / sqrt(fan_in) * min(1.0, sqrt(fan_out / fan_in))
//!
//! **Recommendation**: Make `aprender::nn::init::normal()` and `uniform()` public in aprender fork
//! to enable full constitution compliance. Until then, we use the same Box-Muller transform
//! that aprender uses internally, ensuring statistical equivalence.
//!
//! # References
//!
//! - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
//!   deep feedforward neural networks. AISTATS.
//! - He, K., et al. (2015). Delving deep into rectifiers: Surpassing human-level
//!   performance on ImageNet classification. ICCV.

use aprender::autograd::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Initialize Linear layer weight using Python nanochat's scheme
///
/// Python: std = 1.0 / sqrt(fan_in) * min(1.0, sqrt(fan_out / fan_in))
///
/// This matches the initialization scheme used in the Python reference implementation.
///
/// # Arguments
/// * `in_features` - Number of input features (fan_in)
/// * `out_features` - Number of output features (fan_out)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Weight tensor with shape [out_features, in_features]
///
/// # Aprender API Compliance
///
/// Uses `StdRng` with `SeedableRng::seed_from_u64()` per Principle VII.
/// The Box-Muller transform matches aprender's internal implementation.
/// Once `aprender::nn::init::normal()` is made public, this should be refactored to use it.
pub(crate) fn init_linear_weight(
    in_features: usize,
    out_features: usize,
    seed: Option<u64>,
) -> Tensor {
    let fan_in = in_features as f32;
    let fan_out = out_features as f32;

    // Python formula: std = 1.0 / sqrt(fan_in) * min(1.0, sqrt(fan_out / fan_in))
    let std = (1.0 / fan_in.sqrt()) * (1.0_f32.min((fan_out / fan_in).sqrt()));

    // Use Box-Muller transform for normal distribution (same as aprender uses internally)
    // Uses StdRng with proper seeding per Principle VII
    let numel = out_features * in_features;
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mean = 0.0;

    let data: Vec<f32> = (0..numel)
        .map(|_| {
            let u1: f32 = rng.gen_range(0.0001_f32..1.0_f32);
            let u2: f32 = rng.gen_range(0.0_f32..1.0_f32);
            let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
            mean + std * z
        })
        .collect();

    Tensor::new(&data, &[out_features, in_features])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_linear_weight_shape() {
        let weight = init_linear_weight(10, 20, Some(42));
        assert_eq!(weight.shape(), &[20, 10]);
    }

    #[test]
    fn test_init_linear_weight_reproducible() {
        let weight1 = init_linear_weight(10, 20, Some(123));
        let weight2 = init_linear_weight(10, 20, Some(123));
        assert_eq!(weight1.data(), weight2.data());
    }

    #[test]
    fn test_init_linear_weight_std() {
        // Test that weights have approximately the expected std
        let in_features = 100;
        let out_features = 200;
        let weight = init_linear_weight(in_features, out_features, Some(42));

        let fan_in = in_features as f32;
        let fan_out = out_features as f32;
        let expected_std = (1.0 / fan_in.sqrt()) * (1.0_f32.min((fan_out / fan_in).sqrt()));

        let data = weight.data();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let actual_std = variance.sqrt();

        // Allow 20% tolerance for statistical variation
        assert!(
            (actual_std - expected_std).abs() < expected_std * 0.2,
            "Weight std {actual_std} too far from expected {expected_std}"
        );
    }
}
