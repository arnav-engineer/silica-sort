// Internal profiling binary — accesses crate modules directly (no crate name prefix).
use rand::Rng;
use _silica_sort::learned_sort::learned_sort_f64;

fn main() {
    let mut rng = rand::thread_rng();
    let n = 10_000_000usize;

    let uniform: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let mostly_sorted: Vec<f64> = {
        let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        for _ in 0..(n / 20) {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            v.swap(i, j);
        }
        v
    };
    let low_card: Vec<f64> = (0..n)
        .map(|_| rng.gen_range(0..50) as f64)
        .collect();

    // Warm up
    {
        let mut w = uniform.clone();
        learned_sort_f64(&mut w);
    }

    println!("=== Silica Sort Performance Profile (10M elements) ===\n");

    for round in 0..5 {
        let mb = (n * 8) as f64 / 1e6;

        let mut u = uniform.clone();
        let t = std::time::Instant::now();
        learned_sort_f64(&mut u);
        let e = t.elapsed();
        println!("uniform       round {round}: {:>8.3}ms  ({:.0} MB/s)", e.as_secs_f64() * 1000.0, mb / e.as_secs_f64());

        let mut ms = mostly_sorted.clone();
        let t = std::time::Instant::now();
        learned_sort_f64(&mut ms);
        let e = t.elapsed();
        println!("mostly_sorted round {round}: {:>8.3}ms  ({:.0} MB/s)", e.as_secs_f64() * 1000.0, mb / e.as_secs_f64());

        let mut lc = low_card.clone();
        let t = std::time::Instant::now();
        learned_sort_f64(&mut lc);
        let e = t.elapsed();
        println!("low_card      round {round}: {:>8.3}ms  ({:.0} MB/s)\n", e.as_secs_f64() * 1000.0, mb / e.as_secs_f64());
    }
}
