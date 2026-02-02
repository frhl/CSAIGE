//! Mathematical utility functions.

/// Log-sum-exp for numerically stable addition of log-probabilities.
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Add two log-probabilities: log(exp(a) + exp(b)).
pub fn add_logp(a: f64, b: f64) -> f64 {
    log_sum_exp(a, b)
}

/// Clamp a value to a range.
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Safe log: returns -infinity for x <= 0.
pub fn safe_log(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Safe division: returns 0 if denominator is near zero.
pub fn safe_div(num: f64, den: f64) -> f64 {
    if den.abs() > 1e-30 {
        num / den
    } else {
        0.0
    }
}

/// Compute the sign of a number: -1, 0, or 1.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Weighted mean.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let (sum_wv, sum_w) = values
        .iter()
        .zip(weights.iter())
        .fold((0.0, 0.0), |(sv, sw), (&v, &w)| (sv + w * v, sw + w));
    if sum_w > 0.0 {
        sum_wv / sum_w
    } else {
        0.0
    }
}

/// Weighted variance.
pub fn weighted_variance(values: &[f64], weights: &[f64]) -> f64 {
    let mean = weighted_mean(values, weights);
    let (sum_wv2, sum_w) = values
        .iter()
        .zip(weights.iter())
        .fold((0.0, 0.0), |(sv, sw), (&v, &w)| {
            (sv + w * (v - mean).powi(2), sw + w)
        });
    if sum_w > 0.0 {
        sum_wv2 / sum_w
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp() {
        // Use actual values for log-sum-exp
        let a = (0.3_f64).ln();
        let b = (0.7_f64).ln();
        let result = log_sum_exp(a, b);
        assert!((result.exp() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(1.0, 2.0), 0.5);
        assert_eq!(safe_div(1.0, 0.0), 0.0);
    }

    #[test]
    fn test_weighted_mean() {
        assert!((weighted_mean(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]) - 2.0).abs() < 1e-10);
        assert!((weighted_mean(&[1.0, 3.0], &[3.0, 1.0]) - 1.5).abs() < 1e-10);
    }
}
