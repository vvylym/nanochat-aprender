//! Evaluation binary for running benchmarks

use anyhow::Result;
use clap::Parser;
use nanochat_eval::{evaluate_all, EvaluationReport};
use nanochat_model::load_checkpoint;
use nanochat_tokenizer::Tokenizer;
use std::path::PathBuf;

/// Command-line arguments for evaluation
#[derive(Parser, Debug)]
#[command(name = "nanochat-eval")]
#[command(about = "Evaluate nanochat models on standard benchmarks")]
struct Args {
    /// Path to model checkpoint file
    #[arg(long, short = 'm')]
    checkpoint: PathBuf,

    /// Path to tokenizer file
    #[arg(long, short = 't')]
    tokenizer: PathBuf,

    /// Directory containing benchmark data files
    #[arg(long, short = 'd')]
    data_dir: PathBuf,

    /// Output directory for evaluation results
    #[arg(long, short = 'o', default_value = "./eval_results")]
    output_dir: PathBuf,

    /// Batch size for evaluation
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// Maximum tokens to generate per prompt
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Specific benchmark to run (default: all)
    #[arg(long)]
    benchmark: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load model
    println!("Loading model from {:?}...", args.checkpoint);
    let (model, _metadata) = load_checkpoint(&args.checkpoint)?;
    println!("Model loaded successfully");

    // Load tokenizer
    println!("Loading tokenizer from {:?}...", args.tokenizer);
    let tokenizer = Tokenizer::from_directory(&args.tokenizer)?;
    println!("Tokenizer loaded successfully");

    // Run evaluation
    println!("Running evaluation on benchmarks...");
    let results = if let Some(benchmark_name) = &args.benchmark {
        // Run specific benchmark
        match benchmark_name.as_str() {
            "core" => vec![nanochat_eval::core::evaluate_core(
                &model,
                &tokenizer,
                &args.data_dir,
                args.batch_size,
                args.max_tokens,
            )?],
            "arc" => vec![nanochat_eval::arc::evaluate_arc(
                &model,
                &tokenizer,
                &args.data_dir,
                args.batch_size,
                args.max_tokens,
            )?],
            "gsm8k" => vec![nanochat_eval::gsm8k::evaluate_gsm8k(
                &model,
                &tokenizer,
                &args.data_dir,
                args.batch_size,
                args.max_tokens,
            )?],
            "humaneval" => vec![nanochat_eval::humaneval::evaluate_humaneval(
                &model,
                &tokenizer,
                &args.data_dir,
                args.batch_size,
                args.max_tokens,
            )?],
            "mmlu" => vec![nanochat_eval::mmlu::evaluate_mmlu(
                &model,
                &tokenizer,
                &args.data_dir,
                args.batch_size,
                args.max_tokens,
            )?],
            "chatcore" => vec![nanochat_eval::chatcore::evaluate_chatcore(
                &model,
                &tokenizer,
                &args.data_dir,
                args.batch_size,
                args.max_tokens,
            )?],
            _ => {
                anyhow::bail!(
                    "Unknown benchmark: {}. Available: core, arc, gsm8k, humaneval, mmlu, chatcore",
                    benchmark_name
                );
            }
        }
    } else {
        // Run all benchmarks
        evaluate_all(
            &model,
            &tokenizer,
            &args.data_dir,
            args.batch_size,
            args.max_tokens,
        )?
    };

    // Generate report
    let report = EvaluationReport::generate_report(&results);

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Save report as JSON
    let report_json_path = args.output_dir.join("report.json");
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&report_json_path, report_json)?;
    println!("Report saved to {:?}", report_json_path);

    // Save report as Markdown
    let report_md_path = args.output_dir.join("report.md");
    let report_md = report.to_markdown();
    std::fs::write(&report_md_path, report_md)?;
    println!("Markdown report saved to {:?}", report_md_path);

    // Print summary
    println!("\n=== Evaluation Summary ===");
    println!("Average Score: {:.2}%", report.average_score * 100.0);
    println!("\nBenchmark Results:");
    for result in &report.benchmarks {
        println!(
            "  {}: {:.2}% ({} / {})",
            result.benchmark_name,
            result.score * 100.0,
            result.correct,
            result.total_samples
        );
    }

    Ok(())
}
