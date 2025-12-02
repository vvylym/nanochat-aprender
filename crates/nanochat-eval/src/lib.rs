//! Evaluation and benchmarking for nanochat models
//!
//! This crate provides evaluation suites for:
//! - CORE benchmark
//! - ARC benchmarks
//! - GSM8K benchmark
//! - HumanEval benchmark
//! - MMLU benchmark
//! - ChatCORE benchmark

pub mod arc;
pub mod chatcore;
pub mod core;
pub mod gsm8k;
pub mod humaneval;
pub mod mmlu;
pub mod report;
