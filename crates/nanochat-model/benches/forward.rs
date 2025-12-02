//! Benchmark for forward pass performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nanochat_model::{GPT, GPTConfig};
use aprender::autograd::Tensor;

fn bench_forward_pass(c: &mut Criterion) {
    let config = GPTConfig::default();
    let mut model = GPT::new(config);

    let mut group = c.benchmark_group("forward_pass");
    
    // Benchmark different sequence lengths
    for seq_len in [1, 10, 50, 100, 256].iter() {
        let input = Tensor::zeros(&[1, *seq_len]);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("seq_len_{}", seq_len)),
            &input,
            |b, input| {
                b.iter(|| {
                    let _ = black_box(model.forward_cache(black_box(input), None).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_forward_with_kv_cache(c: &mut Criterion) {
    let config = GPTConfig::default();
    let mut model = GPT::new(config);

    let mut group = c.benchmark_group("forward_with_kv_cache");
    
    // Benchmark autoregressive generation (single token with cache)
    let input = Tensor::zeros(&[1, 1]);
    use nanochat_model::attention::KVCache;
    let mut kv_cache = KVCache::new();
    
    group.bench_function("single_token_with_cache", |b| {
        b.iter(|| {
            let _ = black_box(model.forward_cache(black_box(&input), Some(black_box(&mut kv_cache))).unwrap());
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_forward_pass, bench_forward_with_kv_cache);
criterion_main!(benches);

