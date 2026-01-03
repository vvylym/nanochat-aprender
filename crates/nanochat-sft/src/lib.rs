//! Supervised fine-tuning stage for nanochat instruction following
//!
//! This crate implements the supervised fine-tuning stage for fine-tuning the mid-trained model
//! on instruction-following data.

pub mod config;
pub mod dataloader;
pub mod metrics;
pub mod optimizer;
pub mod train;
