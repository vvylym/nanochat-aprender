//! Unit tests for report generation

use nanochat_eval::report::{BenchmarkResult, EvaluationReport};

#[test]
fn test_benchmark_result_creation() {
    use serde_json::{Map, Value};
    let mut metadata_map = Map::new();
    metadata_map.insert("test".to_string(), Value::String("data".to_string()));
    let metadata = Value::Object(metadata_map);
    let result = BenchmarkResult::new("CORE".to_string(), 100, 75, metadata);

    assert_eq!(result.benchmark_name, "CORE");
    assert_eq!(result.total_samples, 100);
    assert_eq!(result.correct, 75);
    assert_eq!(result.score, 0.75);
}

#[test]
fn test_evaluation_report_generation() {
    use serde_json::{Map, Value};
    let empty_metadata = Value::Object(Map::new());
    let results = vec![
        BenchmarkResult::new("CORE".to_string(), 100, 75, empty_metadata.clone()),
        BenchmarkResult::new("ARC".to_string(), 50, 40, empty_metadata),
    ];

    let report = EvaluationReport::generate_report(&results);

    assert_eq!(report.benchmarks.len(), 2);
    assert_eq!(report.average_score, 0.775); // (0.75 + 0.8) / 2 = 0.775
    assert!(!report.timestamp.is_empty());
}

#[test]
fn test_report_markdown_formatting() {
    use serde_json::{Map, Value};
    let empty_metadata = Value::Object(Map::new());
    let results = vec![BenchmarkResult::new(
        "CORE".to_string(),
        100,
        75,
        empty_metadata,
    )];

    let report = EvaluationReport::generate_report(&results);
    let markdown = report.to_markdown();

    assert!(markdown.contains("# Evaluation Report"));
    assert!(markdown.contains("CORE"));
    assert!(markdown.contains("75"));
    assert!(markdown.contains("100"));
}
