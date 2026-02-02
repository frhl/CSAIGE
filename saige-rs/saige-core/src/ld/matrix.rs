//! LD matrix computation for gene-based tests.
//!
//! Computes the LD (r^2) matrix for a set of variants within a region,
//! used as input for conditional analysis and SKAT-type tests.
//!
//! Reference: SAIGE src/LDmat.cpp

use anyhow::Result;

use saige_linalg::dense::DenseMatrix;

/// Compute the LD correlation matrix for a set of variants.
///
/// Returns an m x m matrix where entry (i,j) is the Pearson
/// correlation between dosage vectors of variants i and j.
pub fn compute_ld_matrix(
    dosage_vectors: &[Vec<f64>],
) -> DenseMatrix {
    let m = dosage_vectors.len();
    if m == 0 {
        return DenseMatrix::zeros(0, 0);
    }

    // Precompute means and standard deviations
    let stats: Vec<(f64, f64)> = dosage_vectors
        .iter()
        .map(|g| {
            let (sum, sum_sq, count) = g.iter().fold((0.0, 0.0, 0), |(s, ss, c), &v| {
                if !v.is_nan() {
                    (s + v, ss + v * v, c + 1)
                } else {
                    (s, ss, c)
                }
            });
            let mean = sum / count as f64;
            let var = sum_sq / count as f64 - mean * mean;
            let sd = var.max(0.0).sqrt();
            (mean, sd)
        })
        .collect();

    let mut ld = DenseMatrix::zeros(m, m);

    for i in 0..m {
        ld.set(i, i, 1.0);
        let (mean_i, sd_i) = stats[i];
        if sd_i < 1e-10 {
            continue;
        }

        for j in (i + 1)..m {
            let (mean_j, sd_j) = stats[j];
            if sd_j < 1e-10 {
                continue;
            }

            // Pearson correlation
            let mut cov = 0.0;
            let mut count = 0;
            for (gi_vec, gj_vec) in dosage_vectors[i].iter().zip(dosage_vectors[j].iter()) {
                let gi = *gi_vec;
                let gj = *gj_vec;
                if !gi.is_nan() && !gj.is_nan() {
                    cov += (gi - mean_i) * (gj - mean_j);
                    count += 1;
                }
            }

            let r = if count > 0 {
                cov / (count as f64 * sd_i * sd_j)
            } else {
                0.0
            };

            ld.set(i, j, r);
            ld.set(j, i, r);
        }
    }

    ld
}

/// Write LD matrix to file in SAIGE format.
///
/// Format: space-delimited, one row per line.
pub fn write_ld_matrix(
    ld: &DenseMatrix,
    variant_ids: &[String],
    path: &std::path::Path,
) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    // Header: variant IDs
    writeln!(f, "{}", variant_ids.join("\t"))?;

    for i in 0..ld.nrows() {
        let row: Vec<String> = (0..ld.ncols())
            .map(|j| format!("{:.6}", ld.get(i, j)))
            .collect();
        writeln!(f, "{}", row.join("\t"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ld_perfect_correlation() {
        let g1 = vec![0.0, 1.0, 2.0, 0.0, 1.0];
        let g2 = vec![0.0, 1.0, 2.0, 0.0, 1.0]; // identical
        let ld = compute_ld_matrix(&[g1, g2]);
        assert!((ld.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((ld.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((ld.get(1, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ld_uncorrelated() {
        let g1 = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let g2 = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let ld = compute_ld_matrix(&[g1, g2]);
        assert!((ld.get(0, 1)).abs() < 0.5);
    }

    #[test]
    fn test_ld_symmetric() {
        let g1 = vec![0.0, 1.0, 2.0];
        let g2 = vec![1.0, 1.0, 0.0];
        let g3 = vec![2.0, 0.0, 1.0];
        let ld = compute_ld_matrix(&[g1, g2, g3]);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (ld.get(i, j) - ld.get(j, i)).abs() < 1e-10,
                    "LD not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }
}
