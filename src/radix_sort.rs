use rayon::prelude::*;

const RADIX_BITS: usize = 8;
const RADIX_SIZE: usize = 1 << RADIX_BITS; // 256
const RADIX_MASK: u64 = (RADIX_SIZE as u64) - 1;
const NUM_PASSES: usize = 8; // 64 / 8
const SIGN_MASK: u64 = 1u64 << 63;
const PARALLEL_THRESHOLD: usize = 200_000;
const SMALL_THRESHOLD: usize = 64;

// ==================== Key Transforms ====================

#[inline(always)]
pub fn f64_bits_to_key(bits: u64) -> u64 {
    let sign = bits >> 63;
    bits ^ (sign.wrapping_neg() | SIGN_MASK)
}

#[inline(always)]
pub fn key_to_f64_bits(key: u64) -> u64 {
    if key & SIGN_MASK == 0 {
        !key
    } else {
        key ^ SIGN_MASK
    }
}

#[inline(always)]
fn digit(value: u64, pass: usize) -> usize {
    ((value >> (pass * RADIX_BITS)) & RADIX_MASK) as usize
}

pub fn f64_as_u64_mut(data: &mut [f64]) -> &mut [u64] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u64, data.len()) }
}

// ==================== Public API ====================

/// Sort f64 slice in-place, allocating its own scratch buffer.
pub fn sort_f64_in_place(data: &mut [f64]) {
    let n = data.len();
    if n <= 1 { return; }
    if n <= SMALL_THRESHOLD {
        pdqsort_f64(data);
        return;
    }

    let keys = f64_as_u64_mut(data);
    transform_keys(keys);

    let mut scratch: Vec<u64> = vec![0u64; n];

    if n >= PARALLEL_THRESHOLD {
        radix_sort_parallel(keys, &mut scratch);
    } else {
        radix_sort_sequential(keys, &mut scratch);
    }

    restore_keys(keys);
}

/// Sort f64 slice using a provided scratch buffer (avoids allocation).
pub fn sort_f64_with_scratch(data: &mut [f64], scratch: &mut [f64]) {
    let n = data.len();
    if n <= 1 { return; }
    if n <= SMALL_THRESHOLD {
        pdqsort_f64(data);
        return;
    }

    let keys = f64_as_u64_mut(data);
    let temp = unsafe { std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u64, n) };

    transform_keys(keys);

    if n >= PARALLEL_THRESHOLD {
        radix_sort_parallel(keys, temp);
    } else {
        radix_sort_sequential(keys, temp);
    }

    restore_keys(keys);
}

/// Fallback for `sort_unstable` that uses unsigned integer comparison for f64.
/// `f64::total_cmp` is slow because it checks for NaN/sign.
/// Using transformed u64 keys maps to simple CMP instructions.
pub fn pdqsort_f64(data: &mut [f64]) {
    if data.len() <= 1 { return; }
    let keys = f64_as_u64_mut(data);
    transform_keys(keys);
    keys.sort_unstable();
    restore_keys(keys);
}

// ==================== Sequential Radix Sort ====================

fn radix_sort_sequential(keys: &mut [u64], scratch: &mut [u64]) {
    let n = keys.len();

    // Pre-compute all 8 histograms in ONE read pass.
    // This lets us skip constant-digit passes without extra reads.
    // Total histogram memory: 8 * 256 * 4 = 8KB → fits entirely in L1.
    let mut histograms = [[0u32; RADIX_SIZE]; NUM_PASSES];
    for &key in keys.iter() {
        unsafe {
            *histograms[0].get_unchecked_mut(digit(key, 0)) += 1;
            *histograms[1].get_unchecked_mut(digit(key, 1)) += 1;
            *histograms[2].get_unchecked_mut(digit(key, 2)) += 1;
            *histograms[3].get_unchecked_mut(digit(key, 3)) += 1;
            *histograms[4].get_unchecked_mut(digit(key, 4)) += 1;
            *histograms[5].get_unchecked_mut(digit(key, 5)) += 1;
            *histograms[6].get_unchecked_mut(digit(key, 6)) += 1;
            *histograms[7].get_unchecked_mut(digit(key, 7)) += 1;
        }
    }

    let keys_ptr = keys.as_mut_ptr();
    let scratch_ptr = scratch.as_mut_ptr();
    let mut src = keys_ptr;
    let mut dst = scratch_ptr;
    let mut in_keys = true;

    for (pass, hist_pass) in histograms.iter().enumerate() {
        if is_single_bucket(hist_pass) {
            continue;
        }

        let mut offsets = [0usize; RADIX_SIZE];
        prefix_sum(hist_pass, &mut offsets);

        let source = unsafe { std::slice::from_raw_parts(src, n) };
        for &value in source {
            let d = digit(value, pass);
            unsafe {
                *dst.add(*offsets.get_unchecked(d)) = value;
                *offsets.get_unchecked_mut(d) += 1;
            }
        }

        std::mem::swap(&mut src, &mut dst);
        in_keys = !in_keys;
    }

    if !in_keys {
        unsafe { std::ptr::copy_nonoverlapping(scratch_ptr, keys_ptr, n); }
    }
}

// ==================== Parallel Radix Sort ====================

fn radix_sort_parallel(keys: &mut [u64], scratch: &mut [u64]) {
    let n = keys.len();
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = n.div_ceil(num_threads);
    let num_chunks = n.div_ceil(chunk_size);

    let keys_ptr = keys.as_mut_ptr();
    let scratch_ptr = scratch.as_mut_ptr();
    let mut src = keys_ptr;
    let mut dst = scratch_ptr;
    let mut in_keys = true;

    // Reusable histogram storage
    let mut local_hists = vec![0u32; num_chunks * RADIX_SIZE];

    for pass in 0..NUM_PASSES {
        let source = unsafe { std::slice::from_raw_parts(src, n) };

        // Step 1: Parallel per-chunk histograms
        local_hists.fill(0);
        source
            .par_chunks(chunk_size)
            .zip(local_hists.par_chunks_mut(RADIX_SIZE))
            .for_each(|(chunk, hist)| {
                for &value in chunk {
                    unsafe {
                        *hist.get_unchecked_mut(digit(value, pass)) += 1;
                    }
                }
            });

        // Step 2: Sum to global histogram, check skip
        let mut global = [0u32; RADIX_SIZE];
        for c in 0..num_chunks {
            let base = c * RADIX_SIZE;
            for b in 0..RADIX_SIZE {
                unsafe {
                    *global.get_unchecked_mut(b) += *local_hists.get_unchecked(base + b);
                }
            }
        }

        if is_single_bucket(&global) {
            continue;
        }

        // Step 3: Global prefix sum + per-thread offsets
        let mut global_offsets = [0usize; RADIX_SIZE];
        prefix_sum(&global, &mut global_offsets);

        let mut thread_offsets = vec![0usize; num_chunks * RADIX_SIZE];
        for (b, &g_off) in global_offsets.iter().enumerate() {
            let mut off = g_off;
            for c in 0..num_chunks {
                let slot = c * RADIX_SIZE + b;
                thread_offsets[slot] = off;
                off += local_hists[slot] as usize;
            }
        }

        // Step 4: Parallel scatter with write-combining buffers.
        // Instead of random writes to 256 positions, we buffer 8 elements
        // per bucket (64 bytes = 1 cache line) and flush as sequential bursts.
        // This converts random writes into sequential burst writes.
        const BUF_SIZE: usize = 8;
        let dst_addr = dst as usize;
        source
            .par_chunks(chunk_size)
            .zip(thread_offsets.par_chunks(RADIX_SIZE))
            .for_each(|(chunk, base_off)| {
                let mut offsets: [usize; RADIX_SIZE] = [0; RADIX_SIZE];
                offsets.copy_from_slice(base_off);
                let ptr = dst_addr as *mut u64;

                // Write-combining buffers: 256 buckets * 8 elements = 16KB (fits in L1)
                let mut buffers = [[0u64; BUF_SIZE]; RADIX_SIZE];
                let mut buf_counts = [0u8; RADIX_SIZE];

                for &value in chunk {
                    let d = digit(value, pass);
                    let cnt = unsafe { *buf_counts.get_unchecked(d) } as usize;
                    unsafe {
                        *buffers.get_unchecked_mut(d).get_unchecked_mut(cnt) = value;
                    }
                    if cnt == BUF_SIZE - 1 {
                        // Buffer full — flush as sequential burst
                        let off = unsafe { *offsets.get_unchecked(d) };
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                buffers.get_unchecked(d).as_ptr(),
                                ptr.add(off),
                                BUF_SIZE,
                            );
                            *offsets.get_unchecked_mut(d) = off + BUF_SIZE;
                            *buf_counts.get_unchecked_mut(d) = 0;
                        }
                    } else {
                        unsafe {
                            *buf_counts.get_unchecked_mut(d) = (cnt + 1) as u8;
                        }
                    }
                }

                // Flush remaining buffered elements
                for (d, (&cnt, &off)) in buf_counts.iter().zip(offsets.iter()).enumerate() {
                    let cnt = cnt as usize;
                    if cnt > 0 {
                        for i in 0..cnt {
                            unsafe { *ptr.add(off + i) = buffers[d][i]; }
                        }
                    }
                }
            });

        std::mem::swap(&mut src, &mut dst);
        in_keys = !in_keys;
    }

    if !in_keys {
        unsafe { std::ptr::copy_nonoverlapping(scratch_ptr, keys_ptr, n); }
    }
}

// ==================== Helpers ====================

#[inline]
fn is_single_bucket(hist: &[u32; RADIX_SIZE]) -> bool {
    let mut seen = 0u32;
    for &c in hist {
        seen += (c > 0) as u32;
        if seen > 1 { return false; }
    }
    seen <= 1
}

#[inline]
fn prefix_sum(hist: &[u32; RADIX_SIZE], offsets: &mut [usize; RADIX_SIZE]) {
    let mut running = 0usize;
    for i in 0..RADIX_SIZE {
        offsets[i] = running;
        running += hist[i] as usize;
    }
}

pub fn transform_keys(keys: &mut [u64]) {
    if keys.len() >= PARALLEL_THRESHOLD {
        keys.par_iter_mut().for_each(|b| *b = f64_bits_to_key(*b));
    } else {
        for b in keys.iter_mut() {
            *b = f64_bits_to_key(*b);
        }
    }
}

pub fn restore_keys(keys: &mut [u64]) {
    if keys.len() >= PARALLEL_THRESHOLD {
        keys.par_iter_mut().for_each(|b| *b = key_to_f64_bits(*b));
    } else {
        for b in keys.iter_mut() {
            *b = key_to_f64_bits(*b);
        }
    }
}

/// Counting sort for arrays with very few unique values (binary, ternary, etc.).
pub fn try_counting_sort(data: &mut [f64], max_unique: usize) -> bool {
    let n = data.len();
    if n < 64 { return false; }

    // Sample 256 strided points to detect low cardinality
    let sample_count = 256.min(n);
    let step = n / sample_count;
    let mut unique_bits: Vec<u64> = Vec::with_capacity(max_unique + 1);

    for i in 0..sample_count {
        let bits = data[i * step].to_bits();
        if !unique_bits.contains(&bits) {
            unique_bits.push(bits);
            if unique_bits.len() > max_unique {
                return false;
            }
        }
    }

    let k = unique_bits.len();

    // Parallel scan to count — use chunks to minimize overhead.
    // If an unknown element is found, we return None to abort.
    use rayon::prelude::*;
    let counts_opt = data.par_chunks(8192)
        .map(|chunk| {
            let mut local_counts = vec![0usize; k];
            for &val in chunk {
                let bits = val.to_bits();
                let mut found = false;
                for j in 0..k {
                    if unique_bits[j] == bits {
                        local_counts[j] += 1;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return None;
                }
            }
            Some(local_counts)
        })
        .reduce(|| Some(vec![0usize; k]), |a_opt, b_opt| {
            match (a_opt, b_opt) {
                (Some(mut a), Some(b)) => {
                    for i in 0..k {
                        a[i] += b[i];
                    }
                    Some(a)
                }
                _ => None
            }
        });

    if let Some(counts) = counts_opt {
        // Sort unique values by sort key
        let mut indexed: Vec<(u64, usize)> = unique_bits.iter().copied().zip(counts).collect();
        indexed.sort_unstable_by_key(|&(bits, _)| f64_bits_to_key(bits));

        // Fill result
        let mut offset = 0;
        for (bits, count) in indexed {
            let val = f64::from_bits(bits);
            data[offset..offset + count].fill(val);
            offset += count;
        }
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::sort_f64_in_place;

    #[test]
    fn sorts_random_values() {
        let mut values = vec![3.0, -2.0, 4.0, 1.0, 0.0, -9.0];
        sort_f64_in_place(&mut values);
        assert_eq!(values, vec![-9.0, -2.0, 0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn sorts_duplicates() {
        let mut values = vec![7.0, 7.0, 1.0, 7.0, -2.0, 1.0];
        sort_f64_in_place(&mut values);
        assert_eq!(values, vec![-2.0, 1.0, 1.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn sorts_large_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut data: Vec<f64> = (0..100_000).map(|_| rng.gen_range(-1e6..1e6)).collect();
        let mut reference = data.clone();
        sort_f64_in_place(&mut data);
        reference.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(data, reference);
    }
}
