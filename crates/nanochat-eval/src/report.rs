//! Report generation for evaluation results

use serde::{Deserialize, Serialize};

/// Result from a single benchmark evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark (e.g., "CORE", "ARC", "GSM8K")
    pub benchmark_name: String,
    /// Total number of samples evaluated
    pub total_samples: usize,
    /// Number of correct predictions
    pub correct: usize,
    /// Accuracy score (correct / total_samples)
    pub score: f32,
    /// Additional metadata (benchmark-specific)
    pub metadata: serde_json::Value,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(
        benchmark_name: String,
        total_samples: usize,
        correct: usize,
        metadata: serde_json::Value,
    ) -> Self {
        let score = if total_samples > 0 {
            correct as f32 / total_samples as f32
        } else {
            0.0
        };

        Self {
            benchmark_name,
            total_samples,
            correct,
            score,
            metadata,
        }
    }
}

/// Comprehensive evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// All benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
    /// Overall average score across all benchmarks
    pub average_score: f32,
    /// Timestamp of evaluation
    pub timestamp: String,
}

impl EvaluationReport {
    /// Generate a report from benchmark results
    pub fn generate_report(results: &[BenchmarkResult]) -> Self {
        let average_score = if !results.is_empty() {
            results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32
        } else {
            0.0
        };

        Self {
            benchmarks: results.to_vec(),
            average_score,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Format report as markdown
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Evaluation Report\n\n");
        md.push_str(&format!("**Timestamp**: {}\n\n", self.timestamp));
        md.push_str(&format!(
            "**Average Score**: {:.2}%\n\n",
            self.average_score * 100.0
        ));
        md.push_str("## Benchmark Results\n\n");
        md.push_str("| Benchmark | Samples | Correct | Score |\n");
        md.push_str("|-----------|---------|---------|-------|\n");

        for result in &self.benchmarks {
            md.push_str(&format!(
                "| {} | {} | {} | {:.2}% |\n",
                result.benchmark_name,
                result.total_samples,
                result.correct,
                result.score * 100.0
            ));
        }

        md
    }
}
