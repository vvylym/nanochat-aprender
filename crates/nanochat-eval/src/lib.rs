//! Evaluation and benchmarking for nanochat models
//!
//! This crate provides evaluation suites for:
//! - CORE benchmark
//! - ARC benchmarks
//! - GSM8K benchmark
//! - HumanEval benchmark
//! - MMLU benchmark
//! - ChatCORE benchmark
//!
//! # Example
//!
//! ```no_run
//! use nanochat_eval::{evaluate_all, EvaluationReport};
//! use nanochat_model::load_checkpoint;
//! use nanochat_tokenizer::Tokenizer;
//!
//! // Load model and tokenizer
//! let (model, _) = load_checkpoint("checkpoint.safetensors")?;
//! let tokenizer = Tokenizer::from_directory("./tokenizer_dir")?;
//!
//! // Run all benchmarks
//! let results = evaluate_all(&model, &tokenizer, std::path::Path::new("./benchmark_data"), 1, 64)?;
//!
//! // Generate report
//! let report = EvaluationReport::generate_report(&results);
//! println!("{}", report.to_markdown());
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod arc;
pub mod chatcore;
pub mod core;
mod generate;
pub mod gsm8k;
pub mod humaneval;
pub mod mmlu;
pub mod report;

// Re-export common types
pub use report::{BenchmarkResult, EvaluationReport};

/// Evaluate model on all available benchmarks
///
/// # Arguments
/// * `model` - The GPT model to evaluate
/// * `tokenizer` - The tokenizer for encoding/decoding
/// * `data_dir` - Directory containing benchmark data files
/// * `batch_size` - Batch size for evaluation
/// * `max_tokens` - Maximum tokens to generate per prompt
///
/// # Returns
/// Vector of benchmark results
pub fn evaluate_all(
    model: &nanochat_model::GPT,
    tokenizer: &nanochat_tokenizer::Tokenizer,
    data_dir: &std::path::Path,
    batch_size: usize,
    max_tokens: usize,
) -> anyhow::Result<Vec<BenchmarkResult>> {
    // Run each benchmark
    let results = vec![
        core::evaluate_core(model, tokenizer, data_dir, batch_size, max_tokens)?,
        arc::evaluate_arc(model, tokenizer, data_dir, batch_size, max_tokens)?,
        gsm8k::evaluate_gsm8k(model, tokenizer, data_dir, batch_size, max_tokens)?,
        humaneval::evaluate_humaneval(model, tokenizer, data_dir, batch_size, max_tokens)?,
        mmlu::evaluate_mmlu(model, tokenizer, data_dir, batch_size, max_tokens)?,
        chatcore::evaluate_chatcore(model, tokenizer, data_dir, batch_size, max_tokens)?,
    ];

    Ok(results)
}
