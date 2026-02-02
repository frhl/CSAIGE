//! Permutation-based tests for validation.
//!
//! Provides permutation p-values as a validation mechanism
//! for the asymptotic score test.

use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Compute a permutation p-value for a test statistic.
///
/// Permutes the residuals `n_perm` times and counts how often
/// the permuted test statistic exceeds the observed.
pub fn permutation_pvalue(
    g: &[f64],
    residuals: &[f64],
    observed_stat: f64,
    n_perm: usize,
    seed: u64,
) -> f64 {
    let n = g.len();
    assert_eq!(residuals.len(), n);

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut n_exceed = 0;

    for _ in 0..n_perm {
        indices.shuffle(&mut rng);

        let perm_stat: f64 = g
            .iter()
            .zip(indices.iter())
            .map(|(&gi, &idx)| gi * residuals[idx])
            .sum::<f64>();
        let perm_stat = perm_stat * perm_stat;

        if perm_stat >= observed_stat {
            n_exceed += 1;
        }
    }

    (n_exceed as f64 + 1.0) / (n_perm as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_null() {
        // Under the null (random genotypes, random residuals), p-value should be ~uniform
        let n = 100;
        let g: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let residuals: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let obs_stat = g
            .iter()
            .zip(residuals.iter())
            .map(|(gi, ri)| gi * ri)
            .sum::<f64>();
        let obs_stat = obs_stat * obs_stat;

        let pval = permutation_pvalue(&g, &residuals, obs_stat, 999, 42);
        assert!(pval > 0.0 && pval <= 1.0);
    }
}
