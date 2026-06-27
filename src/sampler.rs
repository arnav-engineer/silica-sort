pub struct Sampler;

impl Sampler {
    /// Fast deterministic strided sample. No RNG overhead.
    /// Returns a sorted sample for RMI training.
    pub fn extract_sample(data: &[f64], sample_size: usize) -> Vec<f64> {
        if data.len() <= sample_size {
            let mut sample = data.to_vec();
            crate::radix_sort::pdqsort_f64(&mut sample);
            return sample;
        }

        let step = data.len() / sample_size;
        let mut sample = Vec::with_capacity(sample_size);

        for i in 0..sample_size {
            sample.push(data[i * step]);
        }

        crate::radix_sort::pdqsort_f64(&mut sample);
        sample
    }
}
