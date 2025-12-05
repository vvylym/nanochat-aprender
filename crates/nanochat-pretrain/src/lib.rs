//! Pretraining stage for nanochat base language modeling
//!
//! This crate implements the pretraining stage for training the base language model
//! on raw text data.

pub mod config;
pub mod dataloader;
pub mod metrics;
pub mod optimizer;
pub mod train;
