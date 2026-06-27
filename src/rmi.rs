#[derive(Debug, Clone, Copy)]
pub struct LinearModel {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearModel {
    pub fn new(slope: f64, intercept: f64) -> Self {
        Self { slope, intercept }
    }

    /// Optimized training for MonotonicRMI directly on sorted data.
    pub fn train_on_sorted(data: &[f64], num_buckets: usize) -> Self {
        let n = data.len();
        if n < 2 {
            if n == 1 {
                return Self::new(0.0, 0.0);
            }
            return Self::new(0.0, 0.0);
        }

        let nf = n as f64;
        let num_buckets_f = num_buckets as f64;

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &x) in data.iter().enumerate() {
            let y = (i as f64 / nf) * num_buckets_f;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denominator = nf * sum_xx - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return Self::new(0.0, sum_y / nf);
        }

        let mut slope = (nf * sum_xy - sum_x * sum_y) / denominator;
        if slope < 0.0 {
            slope = 0.0;
        }

        let intercept = (sum_y - slope * sum_x) / nf;
        Self::new(slope, intercept)
    }

    /// Predict y for a given x.
    pub fn predict_f64(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

pub struct MonotonicRMI {
    model: LinearModel,
    pub num_buckets: usize,
}

impl MonotonicRMI {
    pub fn train(data: &[f64], num_buckets: usize) -> Self {
        let model = LinearModel::train_on_sorted(data, num_buckets);
        Self { model, num_buckets }
    }

    pub fn predict(&self, key: f64) -> usize {
        let pred = self.model.predict_f64(key);
        (pred as usize).clamp(0, self.num_buckets - 1)
    }
}
