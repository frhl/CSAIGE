//! Cauchy Combination Test (CCT) for combining p-values.
//!
//! The CCT combines p-values using the Cauchy distribution:
//!   T = sum_i w_i * tan((0.5 - p_i) * pi)
//!   p_combined = 0.5 - atan(T / sum(w_i)) / pi
//!
//! This is more robust than Fisher's method for dependent test statistics,
//! which is the typical situation in gene-based tests.
//!
//! Reference: SAIGE src/CCT.cpp, Liu & Xie (2020)

use std::f64::consts::PI;

/// Combine p-values using the Cauchy Combination Test.
///
/// # Arguments
/// - `pvalues`: Vector of p-values to combine
/// - `weights`: Optional weights (default: equal weights)
///
/// # Returns
/// Combined p-value
pub fn cauchy_combination_test(pvalues: &[f64], weights: Option<&[f64]>) -> f64 {
    if pvalues.is_empty() {
        return 1.0;
    }

    let n = pvalues.len();

    // Default equal weights
    let default_weights = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);
    assert_eq!(w.len(), n);

    let w_sum: f64 = w.iter().sum();
    if w_sum <= 0.0 {
        return 1.0;
    }

    let mut t_stat = 0.0;

    for (&p, &wi) in pvalues.iter().zip(w.iter()) {
        if p.is_nan() || wi <= 0.0 {
            continue;
        }

        if p <= 0.0 {
            // Extremely significant -> p_combined ~ 0
            return 0.0;
        }

        if p >= 1.0 {
            // p = 1 -> contribution = tan(-pi/2) = -inf, but we cap it
            t_stat += wi * (-1e15);
            continue;
        }

        // For very small p-values, use the approximation:
        // tan((0.5 - p) * pi) ≈ 1 / (p * pi) for small p
        if p < 1e-15 {
            t_stat += wi / (p * PI);
        } else {
            t_stat += wi * ((0.5 - p) * PI).tan();
        }
    }

    // Combined p-value from Cauchy distribution
    // P(T > t) = 0.5 - atan(t / w_sum) / pi
    let p_combined = 0.5 - (t_stat / w_sum).atan() / PI;

    p_combined.clamp(0.0, 1.0)
}

/// CCT across multiple annotation categories and MAF bins.
///
/// Combines p-values from SKAT/BURDEN/SKAT-O across different
/// annotation categories and MAF thresholds within a gene.
pub fn cct_across_categories(
    category_pvalues: &[(String, f64)], // (category_name, p-value)
) -> f64 {
    let pvalues: Vec<f64> = category_pvalues.iter().map(|(_, p)| *p).collect();
    cauchy_combination_test(&pvalues, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cct_single() {
        let pvals = vec![0.05];
        let combined = cauchy_combination_test(&pvals, None);
        assert!((combined - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cct_two_significant() {
        let pvals = vec![0.001, 0.001];
        let combined = cauchy_combination_test(&pvals, None);
        // Combined should be more significant than individual
        assert!(combined < 0.01, "combined={}", combined);
    }

    #[test]
    fn test_cct_mixed() {
        let pvals = vec![0.001, 0.5, 0.9];
        let combined = cauchy_combination_test(&pvals, None);
        // Should still be significant due to one very small p-value
        assert!(combined < 0.05);
    }

    #[test]
    fn test_cct_all_nonsignificant() {
        let pvals = vec![0.5, 0.6, 0.7, 0.8];
        let combined = cauchy_combination_test(&pvals, None);
        assert!(combined > 0.3);
    }

    #[test]
    fn test_cct_weighted() {
        let pvals = vec![0.01, 0.5];
        let w = vec![10.0, 1.0]; // Heavily weight the significant one
        let combined = cauchy_combination_test(&pvals, Some(&w));
        assert!(combined < 0.05);
    }

    #[test]
    fn test_cct_very_small_pvalue() {
        let pvals = vec![1e-20, 0.5];
        let combined = cauchy_combination_test(&pvals, None);
        assert!(combined < 1e-10);
    }

    #[test]
    fn test_cct_empty() {
        let combined = cauchy_combination_test(&[], None);
        assert_eq!(combined, 1.0);
    }
}
