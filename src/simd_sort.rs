use std::arch::x86_64::*;

/// SIMD-accelerated sort for small arrays.
/// Uses AVX2 sorting networks for ≤16 elements, insertion sort for ≤32,
/// and falls back to pdqsort for larger arrays.
#[inline]
pub fn simd_bucket_sort(data: &mut [f64], has_avx2: bool) {
    let n = data.len();
    match n {
        0..=1 => {},
        2 => sort2(data),
        3 => sort3(data),
        4 => {
            if has_avx2 {
                unsafe { avx2_sort_4(data) };
            } else {
                sort_small(data);
            }
        },
        5..=8 => {
            if has_avx2 {
                unsafe { avx2_sort_8_partial(data) };
            } else {
                sort_small(data);
            }
        },
        9..=16 => {
            if has_avx2 {
                unsafe { avx2_sort_16_partial(data) };
            } else {
                sort_small(data);
            }
        },
        17..=64 => sort_small(data),
        _ => crate::radix_sort::pdqsort_f64(data),
    }
}

// === Scalar sorts for tiny arrays ===

#[inline]
fn sort2(data: &mut [f64]) {
    if data[0] > data[1] { data.swap(0, 1); }
}

#[inline]
fn sort3(data: &mut [f64]) {
    if data[0] > data[1] { data.swap(0, 1); }
    if data[1] > data[2] { data.swap(1, 2); }
    if data[0] > data[1] { data.swap(0, 1); }
}

/// Optimized insertion sort for small arrays (≤64 elements).
/// Extremely fast for nearly-sorted data from RMI buckets.
#[inline]
fn sort_small(data: &mut [f64]) {
    for i in 1..data.len() {
        let val = data[i];
        let mut j = i;
        while j > 0 && data[j - 1] > val {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = val;
    }
}

// === AVX2 SIMD Sorting Networks ===

/// Sort exactly 4 f64 values using AVX2 sorting network.
#[target_feature(enable = "avx2")]
unsafe fn avx2_sort_4(data: &mut [f64]) {
    let mut v = _mm256_loadu_pd(data.as_ptr());

    // Sorting network for 4 elements: 5 compare-and-swap operations
    // Stage 1: compare (0,1) and (2,3)
    let s1 = _mm256_permute_pd(v, 0b0101); // [1, 0, 3, 2]
    let min1 = _mm256_min_pd(v, s1);
    let max1 = _mm256_max_pd(v, s1);
    v = _mm256_blend_pd(min1, max1, 0b1010); // [min(0,1), max(0,1), min(2,3), max(2,3)]

    // Stage 2: compare (0,2) and (1,3)
    let s2 = _mm256_permute2f128_pd(v, v, 0x01); // swap 128-bit lanes
    let min2 = _mm256_min_pd(v, s2);
    let max2 = _mm256_max_pd(v, s2);
    v = _mm256_blend_pd(min2, max2, 0b1100);

    // Stage 3: compare (1,2) — use scalar for cross-lane swap
    _mm256_storeu_pd(data.as_mut_ptr(), v);
    if data[1] > data[2] { data.swap(1, 2); }
}

/// Sort up to 8 f64 values using AVX2 + merge.
#[target_feature(enable = "avx2")]
unsafe fn avx2_sort_8_partial(data: &mut [f64]) {
    let n = data.len();
    // Pad to 8 with +INF, sort, then truncate
    let mut buf = [f64::INFINITY; 8];
    buf[..n].copy_from_slice(data);
    
    // Sort two halves with AVX2
    avx2_sort_4(&mut buf[0..4]);
    avx2_sort_4(&mut buf[4..8]);
    
    // Merge the two sorted halves (bitonic merge network)
    // Step 1: Reverse second half and compare
    let a = _mm256_loadu_pd(buf.as_ptr());
    let b_rev = _mm256_set_pd(buf[4], buf[5], buf[6], buf[7]); // reverse of second half
    let lo = _mm256_min_pd(a, b_rev);
    let hi = _mm256_max_pd(a, b_rev);
    _mm256_storeu_pd(buf.as_mut_ptr(), lo);
    _mm256_storeu_pd(buf.as_mut_ptr().add(4), hi);
    
    // Step 2: Sort each half again
    avx2_sort_4(&mut buf[0..4]);
    avx2_sort_4(&mut buf[4..8]);
    
    data.copy_from_slice(&buf[..n]);
}

/// Sort up to 16 f64 values using AVX2 with merge.
#[target_feature(enable = "avx2")]
unsafe fn avx2_sort_16_partial(data: &mut [f64]) {
    let n = data.len();
    let mut buf = [f64::INFINITY; 16];
    buf[..n].copy_from_slice(data);
    
    // Sort four groups of 4
    avx2_sort_4(&mut buf[0..4]);
    avx2_sort_4(&mut buf[4..8]);
    avx2_sort_4(&mut buf[8..12]);
    avx2_sort_4(&mut buf[12..16]);
    
    // Merge pairs: (0..4, 4..8) and (8..12, 12..16)
    merge_sorted_halves_4(&mut buf[0..8]);
    merge_sorted_halves_4(&mut buf[8..16]);
    
    // Final merge of two sorted 8-element halves
    merge_sorted_halves_8(&mut buf[0..16]);
    
    data.copy_from_slice(&buf[..n]);
}

/// Merge two sorted 4-element halves into one sorted 8-element array.
#[target_feature(enable = "avx2")]
unsafe fn merge_sorted_halves_4(data: &mut [f64]) {
    let a = _mm256_loadu_pd(data.as_ptr());
    let b_rev = _mm256_set_pd(data[4], data[5], data[6], data[7]);
    let lo = _mm256_min_pd(a, b_rev);
    let hi = _mm256_max_pd(a, b_rev);
    _mm256_storeu_pd(data.as_mut_ptr(), lo);
    _mm256_storeu_pd(data.as_mut_ptr().add(4), hi);
    avx2_sort_4(&mut data[0..4]);
    avx2_sort_4(&mut data[4..8]);
}

/// Merge two sorted 8-element halves into one sorted 16-element array.
#[target_feature(enable = "avx2")]
unsafe fn merge_sorted_halves_8(data: &mut [f64]) {
    // Bitonic merge: reverse second half, compare, then recursively fix
    let mut second = [0.0f64; 8];
    second.copy_from_slice(&data[8..16]);
    second.reverse();
    
    for i in 0..8 {
        if data[i] > second[i] {
            std::mem::swap(&mut data[i], &mut second[i]);
        }
    }
    data[8..16].copy_from_slice(&second);
    
    // Sort each half
    merge_sorted_halves_4(&mut data[0..8]);
    merge_sorted_halves_4(&mut data[8..16]);
}
