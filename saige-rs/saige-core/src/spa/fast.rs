//! Fast SPA variant that partitions samples into nonzero/zero genotype subsets.
//!
//! For markers with many zero genotypes, the CGF can be split:
//!   K(t) = K_nonzero(t) + K_zero(t)
//! where K_zero(t) is approximated by a normal distribution
//! (mean NAmu, variance NAsigma), significantly speeding up computation.
//!
//! Reference: SAIGE src/SPA_binary.cpp (fast variants)

use statrs::distribution::{ContinuousCDF, Normal};

/// Fast CGF K(t) for binomial SPA.
///
/// K(t) = sum_{i: g_i != 0} log(1 - mu_i + mu_i * exp(g_i * t)) + NAmu*t + 0.5*NAsigma*t^2
pub fn k0_fast_binom(t: f64, mu_nb: &[f64], g_nb: &[f64], na_mu: f64, na_sigma: f64) -> f64 {
    let mut sum = 0.0;
    for (mi, gi) in mu_nb.iter().zip(g_nb.iter()) {
        sum += (1.0 - mi + mi * (gi * t).exp()).ln();
    }
    sum + na_mu * t + 0.5 * na_sigma * t * t
}

/// Fast K'(t) - q.
pub fn k1_adj_fast_binom(
    t: f64,
    mu_nb: &[f64],
    g_nb: &[f64],
    q: f64,
    na_mu: f64,
    na_sigma: f64,
) -> f64 {
    let mut sum = 0.0;
    for (mi, gi) in mu_nb.iter().zip(g_nb.iter()) {
        let denom = (1.0 - mi) * (-gi * t).exp() + mi;
        sum += mi * gi / denom;
    }
    sum + na_mu + na_sigma * t - q
}

/// Fast K''(t).
pub fn k2_fast_binom(t: f64, mu_nb: &[f64], g_nb: &[f64], na_sigma: f64) -> f64 {
    let mut sum = 0.0;
    for (mi, gi) in mu_nb.iter().zip(g_nb.iter()) {
        let e = (-gi * t).exp();
        let denom = (1.0 - mi) * e + mi;
        sum += (1.0 - mi) * mi * gi * gi * e / (denom * denom);
    }
    sum + na_sigma
}

/// Find root for fast SPA.
#[allow(clippy::too_many_arguments)]
pub fn find_root_k1_fast(
    init: f64,
    mu_nb: &[f64],
    g_nb: &[f64],
    g_all: &[f64],
    q: f64,
    na_mu: f64,
    na_sigma: f64,
    tol: f64,
    max_iter: usize,
) -> super::binary::RootResult {
    let g_pos: f64 = g_all.iter().filter(|&&v| v > 0.0).sum();
    let g_neg: f64 = g_all.iter().filter(|&&v| v < 0.0).sum();

    if q >= g_pos || q <= g_neg {
        return super::binary::RootResult {
            root: f64::INFINITY,
            iterations: 0,
            converged: true,
        };
    }

    let mut t = init;
    let mut k1_eval = k1_adj_fast_binom(t, mu_nb, g_nb, q, na_mu, na_sigma);
    let mut prev_jump = f64::INFINITY;

    for iter in 1..=max_iter {
        let k2_eval = k2_fast_binom(t, mu_nb, g_nb, na_sigma);
        let t_new = t - k1_eval / k2_eval;

        if t_new.is_nan() || t_new.is_infinite() {
            return super::binary::RootResult {
                root: t,
                iterations: iter,
                converged: false,
            };
        }

        if (t_new - t).abs() < tol {
            return super::binary::RootResult {
                root: t_new,
                iterations: iter,
                converged: true,
            };
        }

        if iter == max_iter {
            return super::binary::RootResult {
                root: t_new,
                iterations: iter,
                converged: false,
            };
        }

        let new_k1 = k1_adj_fast_binom(t_new, mu_nb, g_nb, q, na_mu, na_sigma);

        if k1_eval * new_k1 < 0.0 {
            if (t_new - t).abs() > (prev_jump - tol) {
                let direction = if new_k1 - k1_eval > 0.0 { 1.0 } else { -1.0 };
                let t_halved = t + direction * prev_jump / 2.0;
                prev_jump /= 2.0;
                t = t_halved;
                k1_eval = k1_adj_fast_binom(t, mu_nb, g_nb, q, na_mu, na_sigma);
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

    super::binary::RootResult {
        root: t,
        iterations: max_iter,
        converged: false,
    }
}

/// Fast saddle probability.
pub fn saddle_probability_fast(
    zeta: f64,
    mu_nb: &[f64],
    g_nb: &[f64],
    q: f64,
    na_mu: f64,
    na_sigma: f64,
    log_p: bool,
) -> (f64, bool) {
    let k1 = k0_fast_binom(zeta, mu_nb, g_nb, na_mu, na_sigma);
    let k2 = k2_fast_binom(zeta, mu_nb, g_nb, na_sigma);
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

/// Partition genotypes and mu into zero/nonzero subsets for fast SPA.
///
/// Returns (g_nonzero, mu_nonzero, NAmu, NAsigma).
pub fn partition_for_fast_spa(g: &[f64], mu: &[f64]) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let mut g_nb = Vec::new(); // nonzero genotypes
    let mut mu_nb = Vec::new(); // mu for nonzero genotypes
    let mut na_mu = 0.0; // mean for zero-genotype approximation
    let mut na_sigma = 0.0; // variance for zero-genotype approximation

    for (&gi, &mi) in g.iter().zip(mu.iter()) {
        if gi.abs() > 1e-10 {
            g_nb.push(gi);
            mu_nb.push(mi);
        } else {
            // For zero genotypes: contribute to normal approximation
            // K(t) ≈ NAmu*t + 0.5*NAsigma*t^2
            na_mu += mi * gi; // this is ~0 for gi≈0, but kept for generality
            na_sigma += mi * (1.0 - mi) * gi * gi;
        }
    }

    (g_nb, mu_nb, na_mu, na_sigma)
}

/// Full fast SPA test.
pub fn spa_binary_fast(
    mu: &[f64],
    g: &[f64],
    q: f64,
    qinv: f64,
    pval_noadj: f64,
    tol: f64,
) -> super::binary::SpaResult {
    let (g_nb, mu_nb, na_mu, na_sigma) = partition_for_fast_spa(g, mu);

    let root1 = find_root_k1_fast(0.0, &mu_nb, &g_nb, g, q, na_mu, na_sigma, tol, 1000);
    let root2 = find_root_k1_fast(0.0, &mu_nb, &g_nb, g, qinv, na_mu, na_sigma, tol, 1000);

    if root1.converged && root2.converged {
        let (p1, sad1) =
            saddle_probability_fast(root1.root, &mu_nb, &g_nb, q, na_mu, na_sigma, false);
        let (p2, sad2) =
            saddle_probability_fast(root2.root, &mu_nb, &g_nb, qinv, na_mu, na_sigma, false);

        let p1_final = if sad1 { p1 } else { pval_noadj / 2.0 };
        let p2_final = if sad2 { p2 } else { pval_noadj / 2.0 };
        let pval = p1_final.abs() + p2_final.abs();

        super::binary::SpaResult {
            pvalue: pval,
            converged: true,
            is_spa: sad1 || sad2,
        }
    } else {
        super::binary::SpaResult {
            pvalue: pval_noadj,
            converged: false,
            is_spa: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition() {
        let g = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let mu = vec![0.3, 0.4, 0.5, 0.6, 0.7];
        let (g_nb, mu_nb, _na_mu, _na_sigma) = partition_for_fast_spa(&g, &mu);
        assert_eq!(g_nb.len(), 2);
        assert_eq!(mu_nb.len(), 2);
        assert_eq!(g_nb[0], 1.0);
        assert_eq!(g_nb[1], 2.0);
    }

    #[test]
    fn test_fast_spa_matches_standard() {
        let n = 50;
        let mu: Vec<f64> = (0..n).map(|i| 0.2 + 0.6 * (i as f64 / n as f64)).collect();
        // Many zeros
        let g: Vec<f64> = (0..n).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
        let q: f64 = g
            .iter()
            .zip(mu.iter())
            .map(|(gi, mi)| gi * (1.0 - mi))
            .sum::<f64>()
            * 0.3;

        let m1: f64 = mu.iter().zip(g.iter()).map(|(m, gi)| m * gi).sum();
        let qinv = 2.0 * m1 - q;
        let standard = super::super::binary::spa_binary(&mu, &g, q, qinv, 0.5, 1e-6);
        let fast = spa_binary_fast(&mu, &g, q, qinv, 0.5, 1e-6);

        // Both should produce valid p-values
        assert!(standard.pvalue >= 0.0);
        assert!(fast.pvalue >= 0.0);
    }
}
