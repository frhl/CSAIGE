//! PCG solver wrapper for SAIGE's mixed model equations.
//!
//! Wraps saige_linalg::PcgSolver with SAIGE-specific functionality:
//! - On-the-fly GRM-vector products from packed genotypes
//! - Diagonal preconditioning from Sigma

use saige_linalg::decomposition::{PcgSolver, PcgResult};

/// Configuration for the SAIGE PCG solver.
#[derive(Debug, Clone)]
pub struct SaigePcgConfig {
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for SaigePcgConfig {
    fn default() -> Self {
        Self {
            tol: 1e-5,
            max_iter: 500,
        }
    }
}

/// Solve (tau_e * W + tau_g * GRM) * x = b using PCG.
///
/// This is the core linear solve in SAIGE's AI-REML and score test.
/// GRM-vector products are computed on-the-fly without storing the full GRM.
///
/// # Arguments
/// - `tau`: Variance components [tau_e, tau_g]
/// - `w`: Working weights (diagonal of W), length n
/// - `grm_vec`: Function computing GRM * v
/// - `b`: Right-hand side vector
/// - `config`: PCG configuration
pub fn solve_mixed_model_pcg<F>(
    tau: [f64; 2],
    w: &[f64],
    grm_vec: &F,
    b: &[f64],
    config: &SaigePcgConfig,
) -> PcgResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    assert_eq!(w.len(), n);

    let w_owned = w.to_vec();
    let mat_vec = move |v: &[f64]| -> Vec<f64> {
        let grm_v = grm_vec(v);
        v.iter()
            .zip(w_owned.iter())
            .zip(grm_v.iter())
            .map(|((vi, wi), gi)| tau[0] * wi * vi + tau[1] * gi)
            .collect()
    };

    // Diagonal preconditioner
    let w_precond = w.to_vec();
    let precond = move |v: &[f64]| -> Vec<f64> {
        v.iter()
            .zip(w_precond.iter())
            .map(|(vi, wi)| {
                let diag = tau[0] * wi + tau[1];
                if diag.abs() > 1e-30 { vi / diag } else { *vi }
            })
            .collect()
    };

    let pcg = PcgSolver::new(config.tol, config.max_iter);
    pcg.solve(mat_vec, precond, b, None)
}

/// Compute on-the-fly GRM-vector product from genotype dosages.
///
/// GRM = (1/M) * sum_m (g_m - 2*p_m) * (g_m - 2*p_m)' / (2*p_m*(1-p_m))
///
/// GRM * v = (1/M) * sum_m (g_m - 2*p_m) * ((g_m - 2*p_m)' * v) / (2*p_m*(1-p_m))
///
/// This avoids storing the N x N GRM matrix.
pub struct OnTheFlyGrm {
    /// Standardized genotype matrix: each column is (g_m - 2*p_m) / sqrt(2*p_m*(1-p_m))
    /// stored as columns in a flat array (n_samples * n_markers).
    std_genotypes: Vec<f64>,
    n_samples: usize,
    n_markers: usize,
}

impl OnTheFlyGrm {
    /// Create from raw dosage vectors and allele frequencies.
    pub fn new(dosages: &[Vec<f64>], allele_freqs: &[f64]) -> Self {
        let n_markers = dosages.len();
        let n_samples = if n_markers > 0 { dosages[0].len() } else { 0 };
        let mut std_genotypes = vec![0.0; n_samples * n_markers];

        for (m, (geno, &af)) in dosages.iter().zip(allele_freqs.iter()).enumerate() {
            let denom = (2.0 * af * (1.0 - af)).sqrt();
            if denom > 1e-10 {
                let mean = 2.0 * af;
                for i in 0..n_samples {
                    let g = if geno[i].is_nan() { mean } else { geno[i] };
                    std_genotypes[m * n_samples + i] = (g - mean) / denom;
                }
            }
        }

        Self {
            std_genotypes,
            n_samples,
            n_markers,
        }
    }

    /// Compute GRM * v on-the-fly.
    #[allow(clippy::needless_range_loop)]
    pub fn mat_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.n_samples);
        let mut result = vec![0.0; self.n_samples];

        for m in 0..self.n_markers {
            let offset = m * self.n_samples;
            // dot = g_m' * v
            let mut dot = 0.0;
            for i in 0..self.n_samples {
                dot += self.std_genotypes[offset + i] * v[i];
            }
            // result += g_m * dot
            for i in 0..self.n_samples {
                result[i] += self.std_genotypes[offset + i] * dot;
            }
        }

        // Scale by 1/M
        let scale = 1.0 / self.n_markers as f64;
        for r in &mut result {
            *r *= scale;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_the_fly_grm_identity_like() {
        // If all genotypes are identical and standardized, GRM should be close to identity
        let dosages = vec![vec![0.0, 1.0, 2.0, 1.0, 0.0]; 100];
        let afs = vec![0.4; 100]; // allele freq matching the mean dosage

        let grm = OnTheFlyGrm::new(&dosages, &afs);
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let result = grm.mat_vec(&v);

        // GRM[0,0] should be positive
        assert!(result[0] > 0.0, "GRM[0,0] = {}", result[0]);
    }

    #[test]
    fn test_pcg_simple() {
        let tau = [1.0, 0.1];
        let w = vec![1.0; 5];
        let grm = |v: &[f64]| -> Vec<f64> { v.to_vec() }; // identity GRM
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = SaigePcgConfig::default();
        let result = solve_mixed_model_pcg(tau, &w, &grm, &b, &config);

        assert!(result.converged);
        // A = tau_e * I + tau_g * I = 1.1 * I
        // x = b / 1.1
        for (i, (xi, bi)) in result.x.iter().zip(b.iter()).enumerate().take(5) {
            assert!(
                (xi - bi / 1.1).abs() < 1e-5,
                "x[{}]={}, expected {}",
                i, xi, bi / 1.1
            );
        }
    }
}
