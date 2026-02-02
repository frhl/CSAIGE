//! Model serialization and deserialization.
//!
//! Uses bincode for fast, compact binary serialization.
//! Format: magic bytes (SGMD) + version (u32) + bincode payload.
//! Optional JSON sidecar for human inspection.

use anyhow::{bail, Result};
use std::path::Path;

use super::null_model::NullModel;

/// Save a null model to a binary file (.saige.model).
pub fn save_model(model: &NullModel, path: &Path) -> Result<()> {
    let encoded = bincode::serialize(model)?;
    std::fs::write(path, &encoded)?;
    Ok(())
}

/// Load a null model from a binary file (.saige.model).
pub fn load_model(path: &Path) -> Result<NullModel> {
    let data = std::fs::read(path)?;
    let model: NullModel = bincode::deserialize(&data)?;

    // Validate magic bytes
    if model.magic != NullModel::MAGIC {
        bail!(
            "Invalid model file: expected magic bytes {:?}, got {:?}",
            NullModel::MAGIC,
            model.magic
        );
    }

    Ok(model)
}

/// Save a JSON sidecar for debugging (.saige.model.json).
pub fn save_model_json(model: &NullModel, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(model)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Summary of a null model (for display).
pub fn model_summary(model: &NullModel) -> String {
    format!(
        "SAIGE Null Model v{}\n\
         Trait type: {:?}\n\
         Samples: {}\n\
         Covariates: {}\n\
         Tau: [{:.6}, {:.6}]\n\
         Variance ratio: {:.4}\n\
         N_eff: {:.1}\n\
         LOCO: {}\n\
         Sparse GRM: {}",
        model.version,
        model.trait_type,
        model.n_samples,
        model.n_covariates,
        model.tau[0],
        model.tau[1],
        model.variance_ratio.variance_ratio,
        model.n_eff(),
        if model.use_loco { "yes" } else { "no" },
        if model.use_sparse_grm { "yes" } else { "no" },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glmm::link::TraitType;
    use crate::glmm::variance_ratio::VarianceRatioResult;

    #[test]
    fn test_save_load_roundtrip() {
        let vr = VarianceRatioResult {
            variance_ratio: 0.94,
            categorical_vr: Vec::new(),
            n_markers_used: 30,
            per_marker_vr: vec![0.94],
        };

        let model = NullModel::new(
            TraitType::Binary,
            vec!["S1".into(), "S2".into()],
            [1.0, 0.5],
            vec![0.1],
            vec![0.3, 0.7],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            1,
            vec![0.5, 0.5],
            vr,
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.saige.model");

        save_model(&model, &path).unwrap();
        let loaded = load_model(&path).unwrap();

        assert_eq!(loaded.n_samples, 2);
        assert_eq!(loaded.tau, [1.0, 0.5]);
        assert_eq!(loaded.sample_ids, vec!["S1", "S2"]);
        assert!((loaded.variance_ratio.variance_ratio - 0.94).abs() < 1e-10);
    }
}
