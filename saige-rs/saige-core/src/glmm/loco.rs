//! Leave-One-Chromosome-Out (LOCO) procedure.
//!
//! When testing variants on chromosome c, the null model is fit
//! excluding chromosome c from the GRM. This avoids proximal
//! contamination and produces calibrated test statistics.

use std::collections::HashMap;
use anyhow::Result;
use tracing::info;

use saige_linalg::dense::DenseMatrix;

use super::ai_reml::{AiRemlConfig, fit_ai_reml};
use super::link::TraitType;

/// Results from LOCO procedure: one null model per chromosome.
#[derive(Debug, Clone)]
pub struct LocoResults {
    /// Per-chromosome results. Key = chromosome name.
    pub per_chrom: HashMap<String, LocoChromResult>,
}

/// LOCO result for a single chromosome.
#[derive(Debug, Clone)]
pub struct LocoChromResult {
    /// Chromosome excluded from GRM.
    pub chrom: String,
    /// Variance components [tau_e, tau_g].
    pub tau: [f64; 2],
    /// Fitted values mu (from model excluding this chromosome).
    pub mu: Vec<f64>,
    /// Residuals y - mu.
    pub residuals: Vec<f64>,
    /// Working weights.
    pub working_weights: Vec<f64>,
    /// Fixed effects.
    pub alpha: Vec<f64>,
}

/// Run the LOCO procedure.
///
/// For each unique chromosome in the genotype data, fit a null model
/// excluding that chromosome's markers from the GRM.
///
/// # Arguments
/// - `y`: Phenotype vector
/// - `x`: Design matrix
/// - `marker_chroms`: Chromosome assignment for each marker
/// - `marker_dosages`: Genotype dosage vectors for all markers
/// - `marker_afs`: Allele frequencies for all markers
/// - `trait_type`: Trait type
/// - `config`: AI-REML configuration
pub fn run_loco(
    y: &[f64],
    x: &DenseMatrix,
    marker_chroms: &[String],
    marker_dosages: &[Vec<f64>],
    marker_afs: &[f64],
    trait_type: TraitType,
    config: &AiRemlConfig,
) -> Result<LocoResults> {
    let n = y.len();

    // Get unique chromosomes
    let mut chroms: Vec<String> = marker_chroms
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    chroms.sort();

    info!("Running LOCO across {} chromosomes", chroms.len());

    let mut per_chrom = HashMap::new();

    for chrom in &chroms {
        info!("LOCO: fitting model excluding chromosome {}", chrom);

        // Build on-the-fly GRM excluding this chromosome
        let excluded_indices: Vec<usize> = marker_chroms
            .iter()
            .enumerate()
            .filter(|(_, c)| c.as_str() != chrom.as_str())
            .map(|(i, _)| i)
            .collect();

        let excluded_dosages: Vec<&Vec<f64>> = excluded_indices
            .iter()
            .map(|&i| &marker_dosages[i])
            .collect();

        let excluded_afs: Vec<f64> = excluded_indices
            .iter()
            .map(|&i| marker_afs[i])
            .collect();

        // Build GRM-vector product function for this chromosome exclusion
        let n_markers = excluded_dosages.len();
        let n_samples = n;

        // Precompute standardized genotypes
        let mut std_geno = vec![0.0; n_samples * n_markers];
        for (m, (&dosage_vec, &af)) in excluded_dosages.iter().zip(excluded_afs.iter()).enumerate() {
            let denom = (2.0 * af * (1.0 - af)).sqrt();
            if denom > 1e-10 {
                let mean = 2.0 * af;
                for i in 0..n_samples {
                    let g = if dosage_vec[i].is_nan() { mean } else { dosage_vec[i] };
                    std_geno[m * n_samples + i] = (g - mean) / denom;
                }
            }
        }

        let grm_vec = move |v: &[f64]| -> Vec<f64> {
            let mut result = vec![0.0; n_samples];
            for m in 0..n_markers {
                let offset = m * n_samples;
                let mut dot = 0.0;
                for i in 0..n_samples {
                    dot += std_geno[offset + i] * v[i];
                }
                for i in 0..n_samples {
                    result[i] += std_geno[offset + i] * dot;
                }
            }
            let scale = if n_markers > 0 { 1.0 / n_markers as f64 } else { 0.0 };
            for r in &mut result {
                *r *= scale;
            }
            result
        };

        let reml_result = fit_ai_reml(y, x, grm_vec, trait_type, config)?;

        per_chrom.insert(
            chrom.clone(),
            LocoChromResult {
                chrom: chrom.clone(),
                tau: reml_result.tau,
                mu: reml_result.mu,
                residuals: reml_result.residuals,
                working_weights: reml_result.working_weights,
                alpha: reml_result.alpha,
            },
        );
    }

    Ok(LocoResults { per_chrom })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loco_two_chroms() {
        // Simple quantitative model with 2 chromosomes
        let n = 20;
        let y: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let x_data = vec![1.0; n]; // intercept only
        let x = DenseMatrix::from_col_major(n, 1, x_data);

        // 4 markers: 2 on chr1, 2 on chr2
        let marker_chroms = vec!["1".into(), "1".into(), "2".into(), "2".into()];
        let marker_dosages: Vec<Vec<f64>> = (0..4)
            .map(|m| (0..n).map(|i| ((i + m) % 3) as f64).collect())
            .collect();
        let marker_afs = vec![0.3, 0.4, 0.35, 0.45];

        let config = AiRemlConfig {
            max_iter: 5,
            tol: 1e-3,
            pcg_tol: 1e-3,
            pcg_max_iter: 50,
            n_random_vectors: 3,
            use_sparse_grm: false,
            seed: 42,
        };

        let result = run_loco(&y, &x, &marker_chroms, &marker_dosages, &marker_afs,
                              TraitType::Quantitative, &config).unwrap();

        // Should have results for both chromosomes
        assert!(result.per_chrom.contains_key("1"), "Missing chr1 results");
        assert!(result.per_chrom.contains_key("2"), "Missing chr2 results");
        assert_eq!(result.per_chrom.len(), 2);

        // Each result should have proper dimensions
        for (chrom, res) in &result.per_chrom {
            assert_eq!(res.mu.len(), n, "chr{} mu has wrong length", chrom);
            assert_eq!(res.residuals.len(), n, "chr{} residuals has wrong length", chrom);
            assert!(res.tau[0] > 0.0, "chr{} tau_e should be positive", chrom);
        }
    }
}
