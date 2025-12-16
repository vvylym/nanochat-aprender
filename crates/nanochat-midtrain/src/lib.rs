//! Mid-training stage for nanochat conversational fine-tuning
//!
//! This crate implements the mid-training stage for fine-tuning the pretrained model
//! on conversational data.

pub mod config;
pub mod dataloader;
pub mod metrics;
pub mod optimizer;
pub mod train;
