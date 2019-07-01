#[macro_use]
extern crate criterion;

use criterion::Criterion;
//use criterion::black_box;
//use rayon::prelude::*;

fn increment(input: &mut [f32]) {
    input.iter_mut()
        .for_each(|p| *p += 1. );

}
fn add_up(input: &[f32])->f32 {
    input.iter()
        .sum()

}

fn fibonacci() {
    let mut v= vec![0f32; 1_000_000];
    increment(&mut v);
    add_up(&v);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
