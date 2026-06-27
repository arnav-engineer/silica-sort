use crate::radix_sort;
use crate::sampler::Sampler;
use crate::rmi::MonotonicRMI;
use rayon::prelude::*;



const SAMPLE_SIZE: usize = 10_000;
const SMALL_SORT_THRESHOLD: usize = 64;
// 2048 buckets: each bucket ≈ n/2048 elements.
// For 50M → 24K elements/bucket (192KB) — fits in L2 cache.
const NUM_BUCKETS: usize = 2048;

pub fn learned_sort_f64(data: &mut [f64]) {
    let n = data.len();
    if n <= 1 { return; }
    if n <= SMALL_SORT_THRESHOLD {
        crate::radix_sort::pdqsort_f64(data);
        return;
    }

    // Fast-paths
    if radix_sort::try_counting_sort(data, 16) { return; }
    if check_sorted_or_reverse(data) { return; }

    if n < 100_000 {
        // Small arrays: direct sort (avoids partition overhead)
        radix_sort::sort_f64_in_place(data);
        return;
    }

    // === RMI Distribution Sort ===
    let has_avx2 = std::arch::is_x86_feature_detected!("avx2");

    // 1. Train RMI on sample
    let sample = Sampler::extract_sample(data, SAMPLE_SIZE);
    let rmi = MonotonicRMI::train(&sample, NUM_BUCKETS);

    // 2. Partition into 2048 buckets with write-combining + streaming stores
    let (mut scratch, offsets) = parallel_partition_nt(data, &rmi);

    // 3. Sort each bucket in parallel, writing result back into data[]
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let data_ptr = data.as_mut_ptr() as usize;

    (0..NUM_BUCKETS).into_par_iter().for_each(|b| {
        let start = offsets[b];
        let end = offsets[b + 1];
        let len = end - start;
        if len <= 1 { return; }

        let bucket = unsafe {
            std::slice::from_raw_parts_mut((scratch_ptr as *mut f64).add(start), len)
        };
        let dest = unsafe {
            std::slice::from_raw_parts_mut((data_ptr as *mut f64).add(start), len)
        };

        // SIMD-accelerated sorting for bucket data.
        // After RMI partitioning, data within each bucket is roughly sorted.
        // - ≤64 elements: SIMD sorting networks + insertion sort (L1-resident)
        // - ≤100K elements: pdqsort (exploits partial order, in-place, no alloc)
        // - >100K elements: radix sort (for badly-predicted overflow buckets)
        if len <= 64 {
            crate::simd_sort::simd_bucket_sort(bucket, has_avx2);
        } else if len <= 100_000 {
            // pdqsort: optimal for RMI buckets because the data is already
            // roughly sorted. pdqsort detects runs and uses insertion sort
            // for nearly-sorted regions. No scratch allocation needed.
            crate::radix_sort::pdqsort_f64(bucket);
        } else {
            // Overflow bucket: RMI prediction was poor for this region.
            // Fall back to radix sort with dest[] as scratch space.
            radix_sort::sort_f64_with_scratch(bucket, dest);
        }
        dest.copy_from_slice(bucket);
    });
}

pub fn learned_sort_f64_vec(data: &[f64]) -> Vec<f64> {
    let mut out = data.to_vec();
    learned_sort_f64(&mut out);
    out
}

/// Parallel partition with non-temporal (streaming) stores for scatter.
fn parallel_partition_nt(data: &[f64], rmi: &MonotonicRMI) -> (Vec<f64>, Vec<usize>) {
    let n = data.len();
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = n.div_ceil(num_threads);
    let num_chunks = n.div_ceil(chunk_size);

    // Phase 1: Parallel histograms
    let mut local_histograms = vec![0usize; num_chunks * NUM_BUCKETS];
    data.par_chunks(chunk_size)
        .zip(local_histograms.par_chunks_mut(NUM_BUCKETS))
        .for_each(|(chunk, hist)| {
            for &x in chunk {
                unsafe { *hist.get_unchecked_mut(rmi.predict(x)) += 1; }
            }
        });

    // Phase 2: Prefix sum
    let mut global_offsets = vec![0usize; NUM_BUCKETS + 1];
    let mut thread_starts = vec![0usize; num_chunks * NUM_BUCKETS];
    let mut offset = 0usize;
    for b in 0..NUM_BUCKETS {
        global_offsets[b] = offset;
        for c in 0..num_chunks {
            thread_starts[c * NUM_BUCKETS + b] = offset;
            offset += local_histograms[c * NUM_BUCKETS + b];
        }
    }
    global_offsets[NUM_BUCKETS] = n;

    // Phase 3: Scatter with write-combining
    // SAFETY: every element of `data` will be written exactly once by the
    // scatter loop below, so reading from `output` after the loop is safe.
    // We use zeroed() here to satisfy Clippy's uninit_vec lint while keeping
    // the same performance (the zeroing cost is negligible vs. the scatter).
    let mut output: Vec<f64> = vec![0.0f64; n];
    let out_ptr = output.as_mut_ptr() as usize;

    data.par_chunks(chunk_size)
        .enumerate()
        .for_each(|(c_idx, chunk)| {
            let base = c_idx * NUM_BUCKETS;

            let mut cursors = vec![0usize; NUM_BUCKETS];
            for b in 0..NUM_BUCKETS {
                cursors[b] = thread_starts[base + b];
            }

            // Write-combining buffers: NUM_BUCKETS * 2 f64s = 32KB
            // Using 2 elements per buffer (16 bytes = one _mm_stream_pd)
            // to keep total buffer memory reasonable with 2048 buckets.
            const BUF_SIZE: usize = 2;
            let mut buffers = vec![[0.0f64; BUF_SIZE]; NUM_BUCKETS];
            let mut counts = vec![0u8; NUM_BUCKETS];

            let ptr = out_ptr as *mut f64;

            for &x in chunk {
                let b = rmi.predict(x);
                let cnt = unsafe { *counts.get_unchecked(b) } as usize;
                unsafe { *buffers.get_unchecked_mut(b).get_unchecked_mut(cnt) = x; }

                if cnt == BUF_SIZE - 1 {
                    let off = unsafe { *cursors.get_unchecked(b) };
                    let dst = unsafe { ptr.add(off) };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buffers.get_unchecked(b).as_ptr(), dst, BUF_SIZE
                        );
                        *cursors.get_unchecked_mut(b) = off + BUF_SIZE;
                        *counts.get_unchecked_mut(b) = 0;
                    }
                } else {
                    unsafe { *counts.get_unchecked_mut(b) = (cnt + 1) as u8; }
                }
            }

            // Flush remaining
            for b in 0..NUM_BUCKETS {
                let cnt = counts[b] as usize;
                for i in 0..cnt {
                    unsafe { *ptr.add(cursors[b] + i) = buffers[b][i]; }
                }
            }
        });


    (output, global_offsets)
}

fn check_sorted_or_reverse(data: &mut [f64]) -> bool {
    let n = data.len();
    if n < 128 { return false; }
    let step = n / 128;
    let mut sorted = true;
    let mut reverse = true;

    for i in 1..128 {
        let prev = data[(i - 1) * step];
        let curr = data[i * step];
        if curr < prev { sorted = false; }
        if curr > prev { reverse = false; }
        if !sorted && !reverse { return false; }
    }

    if sorted {
        for i in 1..n { if data[i] < data[i - 1] { return false; } }
        return true;
    }
    if reverse {
        for i in 1..n { if data[i] > data[i - 1] { return false; } }
        data.reverse();
        return true;
    }
    false
}
