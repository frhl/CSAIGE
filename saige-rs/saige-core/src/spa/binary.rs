//! Saddlepoint Approximation for binary (binomial) traits.
//!
//! Implements the CGF-based SPA following Dey et al. (2017).
//! The cumulant generating function for the binary score test is:
//!   K(t) = sum_i log(1 - mu_i + mu_i * exp(g_i * t))
//! where mu_i is the fitted probability and g_i is the genotype.
//!
//! Reference: SAIGE src/SPA_binary.cpp

use statrs::distribution::{ContinuousCDF, Normal};

/// CGF K(t) for binomial SPA.
///
/// K(t) = sum_i log(1 - mu_i + mu_i * exp(g_i * t))
pub fn k0_binom(t: f64, mu: &[f64], g: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (mi, gi) in mu.iter().zip(g.iter()) {
        sum += (1.0 - mi + mi * (gi * t).exp()).ln();
    }
    sum
}

/// First derivative K'(t) - q.
///
/// K'(t) = sum_i (mu_i * g_i) / ((1 - mu_i) * exp(-g_i * t) + mu_i)
pub fn k1_adj_binom(t: f64, mu: &[f64], g: &[f64], q: f64) -> f64 {
    let mut sum = 0.0;
    for (mi, gi) in mu.iter().zip(g.iter()) {
        let denom = (1.0 - mi) * (-gi * t).exp() + mi;
        sum += mi * gi / denom;
    }
    sum - q
}

/// Second derivative K''(t).
///
/// K''(t) = sum_i ((1-mu_i) * mu_i * g_i^2 * exp(-g_i*t)) / ((1-mu_i)*exp(-g_i*t) + mu_i)^2
pub fn k2_binom(t: f64, mu: &[f64], g: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (mi, gi) in mu.iter().zip(g.iter()) {
        let e = (-gi * t).exp();
        let denom = (1.0 - mi) * e + mi;
        sum += (1.0 - mi) * mi * gi * gi * e / (denom * denom);
    }
    sum
}

/// Result of root finding for K'(t) = q.
#[derive(Debug)]
pub struct RootResult {
    pub root: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Find root of K'(t) = q using Newton-Raphson with safeguards.
///
/// Matches SAIGE's `getroot_K1_Binom` function.
pub fn find_root_k1(
    init: f64,
    mu: &[f64],
    g: &[f64],
    q: f64,
    tol: f64,
    max_iter: usize,
) -> RootResult {
    // Check if q is outside the range of possible score values
    let g_pos: f64 = g.iter().filter(|&&v| v > 0.0).sum();
    let g_neg: f64 = g.iter().filter(|&&v| v < 0.0).sum();

    if q >= g_pos || q <= g_neg {
        return RootResult {
            root: f64::INFINITY,
            iterations: 0,
            converged: true,
        };
    }

    let mut t = init;
    let mut k1_eval = k1_adj_binom(t, mu, g, q);
    let mut prev_jump = f64::INFINITY;

    for iter in 1..=max_iter {
        let k2_eval = k2_binom(t, mu, g);
        let t_new = t - k1_eval / k2_eval;

        if t_new.is_nan() || t_new.is_infinite() {
            return RootResult {
                root: t,
                iterations: iter,
                converged: false,
            };
        }

        if (t_new - t).abs() < tol {
            return RootResult {
                root: t_new,
                iterations: iter,
                converged: true,
            };
        }

        if iter == max_iter {
            return RootResult {
                root: t_new,
                iterations: iter,
                converged: false,
            };
        }

        let new_k1 = k1_adj_binom(t_new, mu, g, q);

        // Sign change check with halving
        if k1_eval * new_k1 < 0.0 {
            if (t_new - t).abs() > (prev_jump - tol) {
                let direction = if new_k1 - k1_eval > 0.0 { 1.0 } else { -1.0 };
                let t_halved = t + direction * prev_jump / 2.0;
                let _new_k1_halved = k1_adj_binom(t_halved, mu, g, q);
                prev_jump /= 2.0;
                t = t_halved;
                k1_eval = k1_adj_binom(t, mu, g, q);
            } else {
                prev_jump = (t_new - t).abs();
                t = t_new;
                k1_eval = new_k1;
            }
        } else {
            t = t_new;
            k1_eval = new_k1;
        }
    }

    RootResult {
        root: t,
        iterations: max_iter,
        converged: false,
    }
}

/// Compute saddlepoint probability using Lugannani-Rice formula.
///
/// Returns (p_value, is_saddle_successful).
pub fn saddle_probability(zeta: f64, mu: &[f64], g: &[f64], q: f64, log_p: bool) -> (f64, bool) {
    let k1 = k0_binom(zeta, mu, g);
    let k2 = k2_binom(zeta, mu, g);
    let temp1 = zeta * q - k1;

    if !k1.is_finite() || !k2.is_finite() || temp1 < 0.0 || k2 < 0.0 {
        let pval = if log_p { f64::NEG_INFINITY } else { 0.0 };
        return (pval, false);
    }

    let w = zeta.signum() * (2.0 * temp1).sqrt();
    let v = zeta * k2.sqrt();

    if w == 0.0 {
        let pval = if log_p { f64::NEG_INFINITY } else { 0.0 };
        return (pval, false);
    }

    let z_test = w + (1.0 / w) * (v / w).ln();
    let norm = Normal::new(0.0, 1.0).unwrap();

    let pval = if z_test > 0.0 {
        if log_p {
            (1.0 - norm.cdf(z_test)).ln()
        } else {
            1.0 - norm.cdf(z_test)
        }
    } else {
        let p0 = norm.cdf(z_test);
        if log_p {
            -p0.ln()
        } else {
            -p0
        }
    };

    (pval, true)
}

/// Full SPA test for binary trait.
///
/// Computes a two-sided p-value using SPA on both tails.
/// Falls back to the normal-approximation p-value if SPA fails.
///
/// # Arguments
/// - `mu`: Fitted probabilities from null model
/// - `g`: Genotype dosage vector (centered or raw)
/// - `q`: Score statistic (g' * (y - mu))
/// - `pval_noadj`: P-value from normal approximation (fallback)
/// - `tol`: Root finding tolerance (default: 1e-5)
pub fn spa_binary(mu: &[f64], g: &[f64], q: f64, pval_noadj: f64, tol: f64) -> SpaResult {
    let qinv = -q;
    let log_p = false;

    let root1 = find_root_k1(0.0, mu, g, q, tol, 1000);
    let root2 = find_root_k1(0.0, mu, g, qinv, tol, 1000);

    if root1.converged && root2.converged {
        let (p1, sad1) = saddle_probability(root1.root, mu, g, q, log_p);
        let (p2, sad2) = saddle_probability(root2.root, mu, g, qinv, log_p);

        let p1_final = if sad1 { p1 } else { pval_noadj / 2.0 };
        let p2_final = if sad2 { p2 } else { pval_noadj / 2.0 };

        let pval = p1_final.abs() + p2_final.abs();

        SpaResult {
            pvalue: pval,
            converged: true,
            is_spa: sad1 || sad2,
        }
    } else {
        SpaResult {
            pvalue: pval_noadj,
            converged: false,
            is_spa: false,
        }
    }
}

/// Result of SPA computation.
#[derive(Debug, Clone)]
pub struct SpaResult {
    /// P-value (two-sided).
    pub pvalue: f64,
    /// Whether the root finding converged.
    pub converged: bool,
    /// Whether SPA was actually applied (vs fallback).
    pub is_spa: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k0_binom() {
        let mu = vec![0.3, 0.5, 0.7];
        let g = vec![0.0, 1.0, 2.0];
        let k0 = k0_binom(0.0, &mu, &g);
        // K(0) = sum(log(1)) = 0
        assert!((k0 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_k1_at_zero() {
        let mu = vec![0.3, 0.5, 0.7];
        let g = vec![0.0, 1.0, 2.0];
        // K'(0) = sum(mu_i * g_i) = 0*0.3 + 1*0.5 + 2*0.7 = 1.9
        let k1 = k1_adj_binom(0.0, &mu, &g, 0.0);
        assert!((k1 - 1.9).abs() < 1e-10);
    }

    #[test]
    fn test_k2_positive() {
        let mu = vec![0.3, 0.5, 0.7];
        let g = vec![0.0, 1.0, 2.0];
        let k2 = k2_binom(0.0, &mu, &g);
        // K''(t) should be positive
        assert!(k2 > 0.0);
    }

    #[test]
    fn test_root_finding_converges() {
        let mu = vec![0.3, 0.5, 0.7, 0.4, 0.6];
        let g = vec![0.0, 1.0, 2.0, 0.0, 1.0];
        let q = 1.0; // A reasonable score value
        let result = find_root_k1(0.0, &mu, &g, q, 1e-6, 1000);
        assert!(result.converged);
    }

    #[test]
    fn test_spa_binary_returns_valid_pvalue() {
        let n = 50;
        let mu: Vec<f64> = (0..n).map(|i| 0.3 + 0.4 * (i as f64 / n as f64)).collect();
        let g: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let q: f64 = g
            .iter()
            .zip(mu.iter())
            .map(|(gi, mi)| gi * (1.0 - mi))
            .sum::<f64>()
            * 0.5;

        let result = spa_binary(&mu, &g, q, 0.05, 1e-6);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_root_extreme_q() {
        let mu = vec![0.5; 10];
        let g = vec![1.0; 10];
        // q outside range -> root = infinity
        let result = find_root_k1(0.0, &mu, &g, 20.0, 1e-6, 100);
        assert!(result.converged);
        assert!(result.root.is_infinite());
    }
}
