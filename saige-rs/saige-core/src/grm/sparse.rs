//! Sparse GRM construction and manipulation.
//!
//! Creates a sparse GRM by computing the full GRM and then
//! zeroing entries below a relatedness cutoff. This is more
//! memory-efficient for large biobanks with mostly unrelated samples.

use anyhow::Result;
use tracing::info;

use saige_geno::traits::GenotypeReader;
use saige_linalg::sparse::SparseMatrix;

/// Construct a sparse GRM from genotype data.
///
/// 1. Computes the dense GRM
/// 2. Zeros out entries below the relatedness cutoff
/// 3. Returns a sparse matrix
pub fn compute_sparse_grm(
    reader: &mut dyn GenotypeReader,
    min_maf: f64,
    relatedness_cutoff: f64,
) -> Result<(SparseMatrix, usize)> {
    let n = reader.n_samples();
    let m = reader.n_markers();

    info!(
        "Computing sparse GRM: {} samples, {} markers, cutoff={}",
        n, m, relatedness_cutoff
    );

    // First pass: compute GRM values for potentially related pairs
    // For efficiency, we use a block approach
    let (dense_grm, n_markers) = super::dense::compute_dense_grm(reader, min_maf)?;

    // Convert to sparse with threshold
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        for j in i..n {
            let val = dense_grm.get(i, j);
            if i == j || val.abs() >= relatedness_cutoff {
                rows.push(i);
                cols.push(j);
                vals.push(val);
                if i != j {
                    rows.push(j);
                    cols.push(i);
                    vals.push(val);
                }
            }
        }
    }

    let sparse = SparseMatrix::from_triplets(n, n, &rows, &cols, &vals);

    info!(
        "Sparse GRM: {} non-zero entries out of {} total ({:.1}% sparse)",
        sparse.nnz(),
        n * n,
        100.0 * (1.0 - sparse.nnz() as f64 / (n * n) as f64)
    );

    Ok((sparse, n_markers))
}

/// Compute sparse GRM from pre-loaded dosage vectors.
pub fn compute_sparse_grm_from_dosages(
    dosages: &[Vec<f64>],
    allele_freqs: &[f64],
    relatedness_cutoff: f64,
) -> SparseMatrix {
    let dense = super::dense::compute_grm_from_dosages(dosages, allele_freqs);
    let n = dense.nrows();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        for j in i..n {
            let val = dense.get(i, j);
            if i == j || val.abs() >= relatedness_cutoff {
                rows.push(i);
                cols.push(j);
                vals.push(val);
                if i != j {
                    rows.push(j);
                    cols.push(i);
                    vals.push(val);
                }
            }
        }
    }

    SparseMatrix::from_triplets(n, n, &rows, &cols, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_grm_cutoff() {
        let dosages = vec![
            vec![0.0, 2.0, 0.0, 2.0],
            vec![0.0, 2.0, 0.0, 2.0],
            vec![0.0, 2.0, 0.0, 2.0],
        ];
        let afs = vec![0.5, 0.5, 0.5];

        // With a high cutoff, should only keep diagonal and very related pairs
        let sparse = compute_sparse_grm_from_dosages(&dosages, &afs, 0.5);
        assert!(sparse.nnz() <= dosages[0].len() * dosages[0].len());
        assert!(sparse.nnz() >= dosages[0].len()); // at least diagonal
    }
}
