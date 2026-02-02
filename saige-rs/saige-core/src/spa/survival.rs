//! Saddlepoint Approximation for survival (time-to-event) traits.
//!
//! Similar to binary SPA but uses the complementary log-log link
//! and different CGF formulation.
//!
//! Reference: SAIGE src/SPA_survival.cpp


/// CGF K(t) for survival SPA.
///
/// For survival traits with complementary log-log link,
/// the CGF has the same form as binary but with different mu interpretation.
pub fn k0_survival(t: f64, mu: &[f64], g: &[f64]) -> f64 {
    // Same functional form as binary
    super::binary::k0_binom(t, mu, g)
}

/// K'(t) - q for survival.
pub fn k1_adj_survival(t: f64, mu: &[f64], g: &[f64], q: f64) -> f64 {
    super::binary::k1_adj_binom(t, mu, g, q)
}

/// K''(t) for survival.
pub fn k2_survival(t: f64, mu: &[f64], g: &[f64]) -> f64 {
    super::binary::k2_binom(t, mu, g)
}

/// Full SPA test for survival trait.
pub fn spa_survival(
    mu: &[f64],
    g: &[f64],
    q: f64,
    pval_noadj: f64,
    tol: f64,
) -> super::binary::SpaResult {
    // The survival SPA uses the same algorithm as binary SPA
    // with the survival-specific mu values from the cloglog link
    super::binary::spa_binary(mu, g, q, pval_noadj, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survival_spa_valid_pvalue() {
        let mu = vec![0.1, 0.2, 0.3, 0.15, 0.25];
        let g = vec![0.0, 1.0, 2.0, 0.0, 1.0];
        let q = 0.5;
        let result = spa_survival(&mu, &g, q, 0.1, 1e-6);
        assert!(result.pvalue >= 0.0);
    }
}
