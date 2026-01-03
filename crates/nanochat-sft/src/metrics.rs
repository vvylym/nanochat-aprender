//! Training metrics logging for mid-training

use aprender::autograd::Tensor;

/// Training metrics for a single step
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Loss value
    pub loss: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Throughput (tokens per second)
    pub throughput: f32,
    /// Step number
    pub step: usize,
}

/// Metrics logger for training
pub struct MetricsLogger {
    log_interval: usize,
    step: usize,
}

impl MetricsLogger {
    /// Create a new metrics logger
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            step: 0,
        }
    }

    /// Log metrics for a training step
    pub fn log_step(
        &mut self,
        loss: &Tensor,
        learning_rate: f32,
        tokens_processed: usize,
        time_elapsed: f32,
    ) {
        self.step += 1;

        if self.step.is_multiple_of(self.log_interval) {
            let loss_value = loss.item();
            let throughput = if time_elapsed > 0.0 {
                tokens_processed as f32 / time_elapsed
            } else {
                0.0
            };

            let metrics = TrainingMetrics {
                loss: loss_value,
                learning_rate,
                throughput,
                step: self.step,
            };

            self.print_metrics(&metrics);
        }
    }

    /// Print metrics to stdout
    fn print_metrics(&self, metrics: &TrainingMetrics) {
        println!(
            "Step {}: loss={:.6}, lr={:.2e}, throughput={:.2} tokens/s",
            metrics.step, metrics.loss, metrics.learning_rate, metrics.throughput
        );
    }

    /// Get current step number
    pub fn step(&self) -> usize {
        self.step
    }
}
