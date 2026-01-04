//! MMLU benchmark implementation

use crate::generate::generate_greedy;
use crate::report::BenchmarkResult;
use anyhow::Result;
use nanochat_model::GPT;
use nanochat_tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Evaluate model on MMLU knowledge benchmark
///
/// # Arguments
/// * `model` - The GPT model to evaluate
/// * `tokenizer` - The tokenizer for encoding/decoding
/// * `data_dir` - Directory containing MMLU benchmark data (mmlu.jsonl)
/// * `batch_size` - Batch size for evaluation (currently unused)
/// * `max_tokens` - Maximum tokens to generate per prompt
///
/// # Returns
/// Benchmark result with accuracy score
pub fn evaluate_mmlu(
    model: &GPT,
    tokenizer: &Tokenizer,
    data_dir: &Path,
    _batch_size: usize,
    max_tokens: usize,
) -> Result<BenchmarkResult> {
    let jsonl_path = data_dir.join("mmlu.jsonl");
    let file = File::open(&jsonl_path)
        .map_err(|e| anyhow::anyhow!("Failed to open MMLU data file: {}", e))?;
    let reader = BufReader::new(file);

    let mut total_samples = 0;
    let mut correct = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON line: {}", e))?;

        let question = json
            .get("question")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'question' field in JSON"))?;

        let _choices = json
            .get("choices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing 'choices' field in JSON"))?;

        let answer = json
            .get("answer")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'answer' field in JSON"))?;

        // Generate text from question
        let generated = generate_greedy(model, tokenizer, question, max_tokens)?;

        // Check if answer appears in generated text
        if generated.contains(answer) {
            correct += 1;
        }

        total_samples += 1;
    }

    use serde_json::{Map, Value};
    let mut metadata_map = Map::new();
    metadata_map.insert("benchmark".to_string(), Value::String("MMLU".to_string()));
    metadata_map.insert(
        "max_tokens".to_string(),
        Value::Number(serde_json::Number::from(max_tokens)),
    );
    let metadata = Value::Object(metadata_map);

    Ok(BenchmarkResult::new(
        "MMLU".to_string(),
        total_samples,
        correct,
        metadata,
    ))
}
