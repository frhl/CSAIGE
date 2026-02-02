//! Dense GRM construction.
//!
//! Computes GRM = (1/M) * sum_m g_m * g_m'
//! where g_m is the standardized genotype vector for marker m:
//!   g_m = (dosage - 2*p) / sqrt(2*p*(1-p))
//!
//! Uses rayon for parallel computation over marker chunks.

use anyhow::Result;
use rayon::prelude::*;
use saige_geno::traits::GenotypeReader;
use saige_linalg::dense::DenseMatrix;
use tracing::info;

/// Compute the dense GRM from genotype data.
///
/// GRM[i,j] = (1/M) * sum_m g_im * g_jm
/// where g_im = (dosage_im - 2*p_m) / sqrt(2*p_m*(1-p_m))
///
/// This constructs the full N x N matrix. For large N, use the
/// on-the-fly GRM-vector product in `glmm::pcg` instead.
pub fn compute_dense_grm(
    reader: &mut dyn GenotypeReader,
    min_maf: f64,
) -> Result<(DenseMatrix, usize)> {
    let n = reader.n_samples();
    let m = reader.n_markers();

    info!("Computing dense GRM: {} samples x {} markers", n, m);

    // Phase 1: Read all markers sequentially and standardize (I/O bound).
    let mut std_genotypes: Vec<Vec<f64>> = Vec::new();

    for marker_idx in 0..m {
        let data = reader.read_marker(marker_idx as u64)?;

        let af = data.af;
        if af < min_maf || af > 1.0 - min_maf {
            continue;
        }

        let var = 2.0 * af * (1.0 - af);
        if var < 1e-10 {
            continue;
        }

        let sd = var.sqrt();
        let mean = 2.0 * af;

        let std_g: Vec<f64> = data
            .dosages
            .iter()
            .map(|&d| {
                let d = if d.is_nan() { mean } else { d };
                (d - mean) / sd
            })
            .collect();

        std_genotypes.push(std_g);
    }

    let n_markers_used = std_genotypes.len();

    // Phase 2: Parallelize rank-1 accumulation by splitting rows into blocks.
    // Each thread computes a block of rows of the upper triangle of GRM.
    let row_block_size = (n / rayon::current_num_threads().max(1)).max(1);

    // Flat upper-triangle storage: for row i, cols i..n
    let grm_data: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .chunks(row_block_size)
        .map(|row_chunk| {
            let mut block = Vec::new();
            for &i in &row_chunk {
                let mut row_vals = vec![0.0; n - i];
                for std_g in &std_genotypes {
                    let gi = std_g[i];
                    for (k, val) in row_vals.iter_mut().enumerate() {
                        *val += gi * std_g[i + k];
                    }
                }
                block.push(row_vals);
            }
            block
        })
        .flatten()
        .collect();

    // Assemble into DenseMatrix
    let mut grm = DenseMatrix::zeros(n, n);
    if n_markers_used > 0 {
        let scale = 1.0 / n_markers_used as f64;
        for (i, row_vals) in grm_data.iter().enumerate() {
            for (k, &val) in row_vals.iter().enumerate() {
                let j = i + k;
                let scaled = val * scale;
                grm.set(i, j, scaled);
                if i != j {
                    grm.set(j, i, scaled);
                }
            }
        }
    }

    info!(
        "Dense GRM computed using {} markers ({} excluded by MAF filter)",
        n_markers_used,
        m - n_markers_used
    );

    Ok((grm, n_markers_used))
}

/// Compute the GRM from pre-loaded dosage vectors (for testing).
pub fn compute_grm_from_dosages(dosages: &[Vec<f64>], allele_freqs: &[f64]) -> DenseMatrix {
    let m = dosages.len();
    if m == 0 {
        return DenseMatrix::zeros(0, 0);
    }
    let n = dosages[0].len();

    // Standardize all markers
    let std_genotypes: Vec<Vec<f64>> = dosages
        .iter()
        .zip(allele_freqs.iter())
        .filter_map(|(g, &af)| {
            let var = 2.0 * af * (1.0 - af);
            if var < 1e-10 {
                return None;
            }
            let sd = var.sqrt();
            let mean = 2.0 * af;

            Some(
                g.iter()
                    .map(|&d| {
                        let d = if d.is_nan() { mean } else { d };
                        (d - mean) / sd
                    })
                    .collect(),
            )
        })
        .collect();

    let n_used = std_genotypes.len();

    // Parallelize rank-1 accumulation by row blocks
    let row_block_size = (n / rayon::current_num_threads().max(1)).max(1);

    let grm_data: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .chunks(row_block_size)
        .map(|row_chunk| {
            let mut block = Vec::new();
            for &i in &row_chunk {
                let mut row_vals = vec![0.0; n - i];
                for std_g in &std_genotypes {
                    let gi = std_g[i];
                    for (k, val) in row_vals.iter_mut().enumerate() {
                        *val += gi * std_g[i + k];
                    }
                }
                block.push(row_vals);
            }
            block
        })
        .flatten()
        .collect();

    let mut grm = DenseMatrix::zeros(n, n);
    if n_used > 0 {
        let scale = 1.0 / n_used as f64;
        for (i, row_vals) in grm_data.iter().enumerate() {
            for (k, &val) in row_vals.iter().enumerate() {
                let j = i + k;
                let scaled = val * scale;
                grm.set(i, j, scaled);
                if i != j {
                    grm.set(j, i, scaled);
                }
            }
        }
    }

    grm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grm_identity_like() {
        // With many markers and unrelated samples, diagonal should be ~1
        let n = 5;
        let m = 500;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let mut dosages = Vec::new();
        let mut afs = Vec::new();
        for _ in 0..m {
            let af = 0.1 + rng.gen::<f64>() * 0.4;
            let g: Vec<f64> = (0..n)
                .map(|_| {
                    let r: f64 = rng.gen();
                    if r < (1.0 - af).powi(2) {
                        0.0
                    } else if r < (1.0 - af).powi(2) + 2.0 * af * (1.0 - af) {
                        1.0
                    } else {
                        2.0
                    }
                })
                .collect();
            dosages.push(g);
            afs.push(af);
        }

        let grm = compute_grm_from_dosages(&dosages, &afs);
        assert_eq!(grm.nrows(), n);

        // Diagonal should be roughly around 1
        for i in 0..n {
            assert!(
                grm.get(i, i) > 0.0,
                "GRM diagonal should be positive: {}",
                grm.get(i, i)
            );
        }
    }

    #[test]
    fn test_grm_symmetric() {
        let dosages = vec![
            vec![0.0, 1.0, 2.0, 1.0],
            vec![2.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let afs = vec![0.5, 0.5, 0.5];
        let grm = compute_grm_from_dosages(&dosages, &afs);

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (grm.get(i, j) - grm.get(j, i)).abs() < 1e-10,
                    "GRM not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }
}
