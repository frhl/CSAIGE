//! NullModel: the fitted null GLMM, used as input for association testing.
//!
//! Contains all precomputed quantities needed for Step 2:
//! - Fitted values (mu), residuals, working weights
//! - Variance components (tau)
//! - Precomputed matrices (XVX, XVX_inv_XV, X)
//! - Variance ratios
//! - Optional sparse GRM and LOCO results

use serde::{Deserialize, Serialize};

use crate::glmm::link::TraitType;
use crate::glmm::variance_ratio::VarianceRatioResult;

/// The fitted null model, serialized to .saige.model files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullModel {
    /// Magic bytes for validation.
    pub magic: [u8; 4],
    /// Version number for forward compatibility.
    pub version: u32,
    /// Trait type (binary, quantitative, survival).
    pub trait_type: TraitType,
    /// Sample IDs in model order.
    pub sample_ids: Vec<String>,
    /// Number of samples.
    pub n_samples: usize,
    /// Number of covariates (including intercept).
    pub n_covariates: usize,
    /// Variance components [tau_e, tau_g].
    pub tau: [f64; 2],
    /// Fixed effects coefficients (alpha).
    pub alpha: Vec<f64>,
    /// Fitted values (mu).
    pub mu: Vec<f64>,
    /// mu * (1 - mu) for binary, 1 for quantitative.
    pub mu2: Vec<f64>,
    /// Residuals (y - mu).
    pub residuals: Vec<f64>,
    /// Phenotype values (y).
    pub y: Vec<f64>,
    /// Design matrix X as flat col-major vector (n x p).
    pub x_flat: Vec<f64>,
    /// Number of columns in X.
    pub x_ncols: usize,
    /// Precomputed (X'VX)^{-1} * X'V as flat col-major (p x n).
    pub xvx_inv_xv_flat: Vec<f64>,
    /// Variance ratio result.
    pub variance_ratio: VarianceRatioResult,
    /// Whether LOCO was used.
    pub use_loco: bool,
    /// Per-chromosome results (if LOCO).
    pub loco_tau: Vec<(String, [f64; 2])>,
    pub loco_mu: Vec<(String, Vec<f64>)>,
    pub loco_residuals: Vec<(String, Vec<f64>)>,
    /// Whether the model used a sparse GRM.
    pub use_sparse_grm: bool,
    /// Number of markers used in GRM construction.
    pub n_markers_in_grm: usize,
}

impl NullModel {
    /// Magic bytes: "SGMD" (SaiGe MoDel).
    pub const MAGIC: [u8; 4] = [b'S', b'G', b'M', b'D'];
    /// Current model version.
    pub const VERSION: u32 = 1;

    /// Create a new null model.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trait_type: TraitType,
        sample_ids: Vec<String>,
        tau: [f64; 2],
        alpha: Vec<f64>,
        mu: Vec<f64>,
        y: Vec<f64>,
        x_flat: Vec<f64>,
        x_ncols: usize,
        xvx_inv_xv_flat: Vec<f64>,
        variance_ratio: VarianceRatioResult,
    ) -> Self {
        let n = sample_ids.len();
        let mu2 = match trait_type {
            TraitType::Binary | TraitType::Survival => mu.iter().map(|&m| m * (1.0 - m)).collect(),
            TraitType::Quantitative => vec![1.0; n],
        };
        let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(&yi, &mi)| yi - mi).collect();

        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            trait_type,
            n_samples: n,
            n_covariates: x_ncols,
            sample_ids,
            tau,
            alpha,
            mu,
            mu2,
            residuals,
            y,
            x_flat,
            x_ncols,
            xvx_inv_xv_flat,
            variance_ratio,
            use_loco: false,
            loco_tau: Vec::new(),
            loco_mu: Vec::new(),
            loco_residuals: Vec::new(),
            use_sparse_grm: false,
            n_markers_in_grm: 0,
        }
    }

    /// Get the effective number of cases (for binary traits).
    pub fn n_eff(&self) -> f64 {
        match self.trait_type {
            TraitType::Binary => {
                let n_cases = self.y.iter().filter(|&&y| y > 0.5).count() as f64;
                let n_controls = self.y.iter().filter(|&&y| y < 0.5).count() as f64;
                if n_cases > 0.0 && n_controls > 0.0 {
                    4.0 / (1.0 / n_cases + 1.0 / n_controls)
                } else {
                    self.n_samples as f64
                }
            }
            _ => self.n_samples as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glmm::variance_ratio::VarianceRatioResult;

    #[test]
    fn test_null_model_creation() {
        let vr = VarianceRatioResult {
            variance_ratio: 0.94,
            categorical_vr: Vec::new(),
            n_markers_used: 30,
            per_marker_vr: vec![0.94],
        };

        let model = NullModel::new(
            TraitType::Binary,
            vec!["S1".into(), "S2".into(), "S3".into()],
            [1.0, 0.5],
            vec![0.1],
            vec![0.3, 0.5, 0.7],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            1,
            vec![0.33, 0.33, 0.33],
            vr,
        );

        assert_eq!(model.n_samples, 3);
        assert_eq!(model.magic, NullModel::MAGIC);
        assert!((model.mu2[0] - 0.21).abs() < 1e-10);
        assert!((model.residuals[0] - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_n_eff() {
        let vr = VarianceRatioResult {
            variance_ratio: 1.0,
            categorical_vr: Vec::new(),
            n_markers_used: 0,
            per_marker_vr: Vec::new(),
        };

        let model = NullModel::new(
            TraitType::Binary,
            vec!["S1".into(), "S2".into(), "S3".into(), "S4".into()],
            [1.0, 1.0],
            vec![0.0],
            vec![0.5; 4],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0; 4],
            1,
            vec![0.25; 4],
            vr,
        );

        // n_eff = 4 / (1/2 + 1/2) = 4
        assert!((model.n_eff() - 4.0).abs() < 1e-10);
    }
}
