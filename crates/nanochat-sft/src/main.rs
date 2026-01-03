//! Supervised fine-tuning stage for nanochat instruction following
//!
//! This binary fine-tunes the mid-trained model for instruction-following tasks.
//!
//! # Usage
//!
//! ```bash
//! nanochat-sft \
//!   --config config.json \
//!   --base-model ./checkpoints/mid-trained.safetensors \
//!   --data-dir ./instruction_data \
//!   --output-dir ./checkpoints \
//!   [--resume checkpoint.safetensors] \
//!   [--workers 4] \
//!   [--log-interval 100] \
//!   [--checkpoint-interval 1000]
//! ```
//!
//! # Tokenizer Usage
//!
//! Uses `nanochat_tokenizer::Tokenizer` which wraps `aprender::text::tokenize::BpeTokenizer`.
//! Ensure vocabulary size matches model config using `GPTConfig::with_tokenizer_vocab_size()` or
//! `validate_tokenizer_model_compatibility()` from `nanochat_model`.

use anyhow::{Context, Result};
use clap::Parser;
use nanochat_model::{checkpoint::load_checkpoint, GPTConfig};
use nanochat_sft::{
    config::TrainingConfigFile,
    optimizer::OptimizerConfig,
    train::{train, TrainingConfig},
};
use nanochat_tokenizer::Tokenizer;
use std::path::PathBuf;

/// Supervised fine-tuning stage for nanochat instruction following
#[derive(Parser, Debug)]
#[command(name = "nanochat-sft")]
#[command(
    about = "Supervised fine-tuning stage for nanochat instruction following",
    long_about = None
)]
struct Args {
    /// Path to training configuration file
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Path to mid-trained base model checkpoint (required, SafeTensors format)
    #[arg(long, value_name = "PATH", required = true)]
    base_model: PathBuf,

    /// Directory containing instruction training data (JSONL files)
    #[arg(long, value_name = "PATH", required = true)]
    data_dir: PathBuf,

    /// Directory for checkpoints and outputs
    #[arg(long, value_name = "PATH", required = true)]
    output_dir: PathBuf,

    /// Path to checkpoint to resume from (SafeTensors format)
    #[arg(long, value_name = "PATH")]
    resume: Option<PathBuf>,

    /// Number of data loading workers
    #[arg(long, default_value = "4")]
    workers: usize,

    /// Steps between logging metrics
    #[arg(long, default_value = "100")]
    log_interval: usize,

    /// Steps between checkpoint saves
    #[arg(long, default_value = "1000")]
    checkpoint_interval: usize,

    /// Suppress progress output
    #[arg(long)]
    quiet: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output_dir))?;

    // Load configuration from file or use defaults
    let config_file = if let Some(config_path) = &args.config {
        TrainingConfigFile::from_file(config_path).context("Failed to load config file")?
    } else {
        // Use defaults if no config provided
        TrainingConfigFile::default()
    };

    // Load base model from checkpoint
    let _model_config = GPTConfig {
        vocab_size: config_file.model.vocab_size,
        n_layer: config_file.model.n_layer,
        n_head: config_file.model.n_head,
        n_kv_head: config_file.model.n_kv_head,
        n_embd: config_file.model.n_embd,
        sequence_len: config_file.model.sequence_len,
        dropout: Some(0.0),
        seed: None,
    };

    // Load base model from checkpoint
    let (mut model, _metadata) = load_checkpoint(&args.base_model)
        .with_context(|| format!("Failed to load base model from: {:?}", args.base_model))?;

    // Load tokenizer (should match the one used for pretraining and mid-training)
    let tokenizer = Tokenizer::from_directory(&args.data_dir)
        .or_else(|_| {
            // If tokenizer not found in data_dir, try to load from base model directory
            let base_dir = args.base_model.parent().unwrap_or(&args.data_dir);
            Tokenizer::from_directory(base_dir)
        })
        .context("Failed to load tokenizer. Ensure tokenizer.json exists in data directory or base model directory")?;

    // Validate tokenizer-model compatibility
    let vocab_size = tokenizer.vocab_size();
    model
        .config()
        .validate_vocab_size(vocab_size)
        .map_err(|e| anyhow::anyhow!("Tokenizer-model incompatibility: {}", e))?;

    // Create optimizer config
    let optimizer_config = OptimizerConfig {
        learning_rate: config_file.optimizer.learning_rate,
        weight_decay: config_file.optimizer.weight_decay,
        beta1: config_file.optimizer.beta1,
        beta2: config_file.optimizer.beta2,
        eps: config_file.optimizer.eps,
        warmup_steps: config_file.optimizer.warmup_steps,
        max_steps: config_file.optimizer.max_steps,
        min_lr: config_file.optimizer.min_lr,
        warmup_ratio: config_file.optimizer.warmup_ratio,
        warmdown_ratio: config_file.optimizer.warmdown_ratio,
        final_lr_frac: config_file.optimizer.final_lr_frac,
    };

    // Create training configuration from loaded config
    let training_config = TrainingConfig {
        batch_size: config_file.training.batch_size,
        seq_len: config_file.training.seq_len,
        gradient_accumulation_steps: config_file.training.gradient_accumulation_steps,
        max_steps: config_file.training.max_steps,
        save_interval: args.checkpoint_interval.max(config_file.training.save_interval),
        log_interval: args.log_interval.max(config_file.training.log_interval),
        grad_clip: config_file.training.grad_clip,
        eval_interval: config_file.training.eval_interval,
        eval_steps: config_file.training.eval_tokens
            / (config_file.training.batch_size * config_file.training.seq_len),
        seed: config_file.training.seed,
    };

    // Run training
    train(
        &mut model,
        &tokenizer,
        &args.data_dir,
        &args.output_dir,
        &training_config,
        Some(optimizer_config),
        args.resume.as_deref(),
    )
    .context("Training failed")?;

    if !args.quiet {
        println!("Supervised fine-tuning completed successfully!");
    }

    Ok(())
}
