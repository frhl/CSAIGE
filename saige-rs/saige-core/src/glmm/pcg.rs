//! PCG solver wrapper for SAIGE's mixed model equations.
//!
//! Wraps saige_linalg::PcgSolver with SAIGE-specific functionality:
//! - On-the-fly GRM-vector products from packed genotypes
//! - Diagonal preconditioning from Sigma

use rayon::prelude::*;
use saige_linalg::decomposition::{PcgResult, PcgSolver};

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
                if diag.abs() > 1e-30 {
                    vi / diag
                } else {
                    *vi
                }
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
///
/// Uses f32 internally to match R SAIGE's `arma::fvec`/`arma::fmat` precision.
/// R SAIGE performs ALL GRM operations in float32, so using f32 here produces
/// GRM products that match R's results, leading to closer tau_g estimates.
pub struct OnTheFlyGrm {
    /// Standardized genotype matrix in f32: each column is (g_m - 2*p_m) / sqrt(2*p_m*(1-p_m))
    /// stored as columns in a flat array (n_samples * n_markers).
    std_genotypes: Vec<f32>,
    n_samples: usize,
    n_markers: usize,
}

impl OnTheFlyGrm {
    /// Create from raw dosage vectors and allele frequencies.
    pub fn new(dosages: &[Vec<f64>], allele_freqs: &[f64]) -> Self {
        let n_markers = dosages.len();
        let n_samples = if n_markers > 0 { dosages[0].len() } else { 0 };
        let mut std_genotypes = vec![0.0f32; n_samples * n_markers];

        for (m, (geno, &af)) in dosages.iter().zip(allele_freqs.iter()).enumerate() {
            let denom = (2.0 * af * (1.0 - af)).sqrt();
            if denom > 1e-10 {
                let mean = 2.0 * af;
                for i in 0..n_samples {
                    let g = if geno[i].is_nan() { mean } else { geno[i] };
                    std_genotypes[m * n_samples + i] = ((g - mean) / denom) as f32;
                }
            }
        }

        Self {
            std_genotypes,
            n_samples,
            n_markers,
        }
    }

    /// Compute GRM * v on-the-fly using rayon parallelism.
    ///
    /// Uses f32 for the inner computation to match R SAIGE's float32 precision.
    /// Input and output are f64 for compatibility with the rest of the pipeline.
    pub fn mat_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.n_samples);
        let n = self.n_samples;

        if self.n_markers == 0 {
            return vec![0.0; n];
        }

        // Cast input vector to f32 to match R SAIGE's float32 operations
        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();

        // Chunk the flat genotype array by marker columns (each column = n elements).
        // Each chunk is a contiguous slice of one or more marker columns.
        let chunk_size = {
            let n_threads = rayon::current_num_threads().max(1);
            let markers_per_thread = self.n_markers / n_threads;
            markers_per_thread.max(64) * n
        };

        let scale = 1.0f32 / self.n_markers as f32;

        let result = self
            .std_genotypes
            .par_chunks(chunk_size)
            .fold(
                || vec![0.0f32; n],
                |mut acc, geno_chunk| {
                    let n_markers_in_chunk = geno_chunk.len() / n;
                    for m in 0..n_markers_in_chunk {
                        let col = &geno_chunk[m * n..(m + 1) * n];
                        // dot = g_m' * v (f32)
                        let dot: f32 = col.iter().zip(v_f32.iter()).map(|(g, vi)| g * vi).sum();
                        // acc += g_m * dot (f32)
                        for (a, g) in acc.iter_mut().zip(col.iter()) {
                            *a += g * dot;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; n],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            );

        // Scale and convert back to f64
        result.into_iter().map(|r| (r * scale) as f64).collect()
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
                i,
                xi,
                bi / 1.1
            );
        }
    }
}
