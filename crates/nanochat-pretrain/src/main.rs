//! Pretraining stage for nanochat base language modeling
//!
//! This binary trains the base language model on raw text data.
//!
//! # Usage
//!
//! ```bash
//! nanochat-pretrain \
//!   --config config.json \
//!   --data-dir ./data \
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
use nanochat_model::{GPTConfig, GPT};
use nanochat_pretrain::{
    config::TrainingConfigFile,
    dataloader::DataLoader,
    optimizer::OptimizerConfig,
    train::{resume_from_checkpoint, train, TrainingConfig},
};
use nanochat_tokenizer::Tokenizer;
use std::path::PathBuf;

/// Pretraining stage for nanochat base language modeling
#[derive(Parser, Debug)]
#[command(name = "nanochat-pretrain")]
#[command(about = "Pretraining stage for nanochat base language modeling", long_about = None)]
struct Args {
    /// Path to training configuration file
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Directory containing training data shards
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

    // Load or train tokenizer (needed for both new and resumed training)
    let tokenizer = create_or_load_tokenizer(&args.data_dir, config_file.model.vocab_size)?;

    // Create optimizer config (needed for both new and resumed training)
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

    // Create training data loader (needed for both new and resumed training)
    let mut dataloader = DataLoader::new(
        &args.data_dir,
        tokenizer.clone(),
        config_file.training.batch_size,
        config_file.training.seq_len,
        args.workers,
    )
    .context("Failed to create data loader")?;

    // Create or load model
    let model = if let Some(resume_path) = &args.resume {
        // Create a temporary model for loading (will be replaced by checkpoint)
        let model_config = GPTConfig {
            vocab_size: config_file.model.vocab_size,
            n_layer: config_file.model.n_layer,
            n_head: config_file.model.n_head,
            n_kv_head: config_file.model.n_kv_head,
            n_embd: config_file.model.n_embd,
            sequence_len: config_file.model.sequence_len,
            dropout: Some(0.0),
            seed: None,
        };
        let mut temp_model = GPT::new(model_config);

        // Resume from checkpoint - this will load model weights and restore dataloader state
        resume_from_checkpoint(
            resume_path,
            optimizer_config.clone(),
            &mut dataloader,
            &mut temp_model,
        )
        .with_context(|| format!("Failed to resume from checkpoint: {:?}", resume_path))?;
        temp_model
    } else {
        // Create new model from config
        let model_config = GPTConfig {
            vocab_size: config_file.model.vocab_size,
            n_layer: config_file.model.n_layer,
            n_head: config_file.model.n_head,
            n_kv_head: config_file.model.n_kv_head,
            n_embd: config_file.model.n_embd,
            sequence_len: config_file.model.sequence_len,
            dropout: Some(0.0), // TODO: Add to config if needed
            seed: None,         // Use default seed
        };
        GPT::new(model_config)
    };

    // Validate tokenizer-model compatibility
    let vocab_size = tokenizer.vocab_size();
    model
        .config()
        .validate_vocab_size(vocab_size)
        .map_err(|e| anyhow::anyhow!("Tokenizer-model incompatibility: {}", e))?;

    // Create validation data loader (if validation data exists)
    let val_data_dir = args.data_dir.join("val");
    let mut val_dataloader_opt = if val_data_dir.exists() {
        Some(
            DataLoader::new(
                &val_data_dir,
                tokenizer,
                config_file.training.batch_size,
                config_file.training.seq_len,
                args.workers,
            )
            .context("Failed to create validation data loader")?,
        )
    } else {
        None
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
    };

    // Run training
    train(
        model,
        dataloader,
        training_config,
        optimizer_config,
        &args.output_dir,
        val_dataloader_opt.as_mut(),
    )
    .context("Training failed")?;

    if !args.quiet {
        println!("Training completed successfully!");
    }

    Ok(())
}

/// Create or load tokenizer from data directory
///
/// # Arguments
/// * `data_dir` - Directory containing data files
/// * `vocab_size` - Target vocabulary size from config
///
/// # Returns
/// Loaded or newly trained tokenizer
fn create_or_load_tokenizer(data_dir: &PathBuf, vocab_size: usize) -> Result<Tokenizer> {
    // Check if tokenizer already exists
    let tokenizer_path = data_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        // Load existing tokenizer
        Tokenizer::from_directory(data_dir)
            .with_context(|| format!("Failed to load tokenizer from: {:?}", data_dir))
    } else {
        // Train new tokenizer from data files
        let special_tokens = vec![
            "<|bos|>".to_string(),
            "<|eos|>".to_string(),
            "<|pad|>".to_string(),
        ];
        Tokenizer::train_from_directory(data_dir, vocab_size, Some(special_tokens))
            .context("Failed to train tokenizer from data files")
    }
}
