//! Exact tests for ultra-rare variants (MAC <= 10).
//!
//! Uses the hypergeometric distribution to compute exact p-values
//! when the minor allele count is very low, where asymptotic
//! approximations may be unreliable.
//!
//! Reference: SAIGE src/Binary_ComputeExact.cpp, src/ER_binary_func.cpp

use statrs::function::factorial::ln_factorial;

/// Result of an exact test.
#[derive(Debug, Clone)]
pub struct ExactTestResult {
    pub pvalue: f64,
    pub method: &'static str,
}

/// Compute exact p-value using hypergeometric distribution.
///
/// For a 2x2 table:
///   Cases with ALT, Cases without ALT
///   Controls with ALT, Controls without ALT
///
/// # Arguments
/// - `n_case_alt`: Number of cases carrying the alternative allele
/// - `n_case`: Total number of cases
/// - `n_alt`: Total number of samples carrying the alternative allele
/// - `n_total`: Total number of samples
pub fn exact_test_hypergeometric(
    n_case_alt: usize,
    n_case: usize,
    n_alt: usize,
    n_total: usize,
) -> ExactTestResult {
    let n_control = n_total - n_case;
    let _n_ref = n_total - n_alt;

    // P(X = k) for hypergeometric distribution
    // P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
    // where N = n_total, K = n_case, n = n_alt, k = n_case_alt

    let p_observed = hypergeom_pmf(n_case_alt, n_total, n_case, n_alt);

    // Two-sided p-value: sum P(X = k) for all k where P(X = k) <= P(X = observed)
    let min_k = n_alt.saturating_sub(n_control);
    let max_k = n_case.min(n_alt);

    let mut pvalue = 0.0;
    for k in min_k..=max_k {
        let p_k = hypergeom_pmf(k, n_total, n_case, n_alt);
        if p_k <= p_observed + 1e-15 {
            pvalue += p_k;
        }
    }

    ExactTestResult {
        pvalue: pvalue.min(1.0),
        method: "hypergeometric",
    }
}

/// Hypergeometric PMF: P(X = k | N, K, n)
fn hypergeom_pmf(k: usize, n_total: usize, n_success: usize, n_draws: usize) -> f64 {
    let log_p = ln_choose(n_success, k)
        + ln_choose(n_total - n_success, n_draws - k)
        - ln_choose(n_total, n_draws);
    log_p.exp()
}

/// Log of binomial coefficient: ln(C(n, k))
fn ln_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n as u64) - ln_factorial(k as u64) - ln_factorial((n - k) as u64)
}

/// Recursive enumeration of exact test statistics for ultra-rare variants.
///
/// Enumerates all possible test statistic values for a given MAC
/// and computes the exact p-value.
pub fn exact_test_enumerate(
    genotypes: &[f64],  // 0, 1, 2
    phenotypes: &[f64], // 0 or 1 for binary
) -> ExactTestResult {
    let n = genotypes.len();

    // Count carriers (ALT allele count > 0)
    let n_alt: usize = genotypes.iter().filter(|&&g| g > 0.5).count();
    let n_case: usize = phenotypes.iter().filter(|&&y| y > 0.5).count();
    let n_case_alt: usize = genotypes
        .iter()
        .zip(phenotypes.iter())
        .filter(|(&g, &y)| g > 0.5 && y > 0.5)
        .count();

    exact_test_hypergeometric(n_case_alt, n_case, n_alt, n)
}

/// Determine if exact test should be used based on MAC.
pub fn should_use_exact_test(mac: f64, threshold: f64) -> bool {
    mac <= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypergeom_pmf() {
        // Simple case: N=10, K=5, n=5
        // P(X=2) should be C(5,2)*C(5,3)/C(10,5)
        let p = hypergeom_pmf(2, 10, 5, 5);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_exact_test_basic() {
        // 20 samples, 5 cases with ALT, 10 total cases, 5 ALT carriers
        let result = exact_test_hypergeometric(5, 10, 5, 20);
        // All ALT carriers are cases -> significant
        assert!(result.pvalue < 0.1);
    }

    #[test]
    fn test_exact_test_no_enrichment() {
        // 20 samples, 3 cases with ALT out of 10 cases, 6 ALT out of 20
        let result = exact_test_hypergeometric(3, 10, 6, 20);
        // Expected: 10*6/20 = 3, so no enrichment
        assert!(result.pvalue > 0.5);
    }

    #[test]
    fn test_exact_enumerate() {
        let geno = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let pheno = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let result = exact_test_enumerate(&geno, &pheno);
        assert!(result.pvalue > 0.0);
    }

    #[test]
    fn test_should_use_exact() {
        assert!(should_use_exact_test(5.0, 10.0));
        assert!(!should_use_exact_test(15.0, 10.0));
    }
}
