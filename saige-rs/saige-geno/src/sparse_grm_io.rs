//! Sparse GRM reader/writer in MatrixMarket-like format.
//!
//! SAIGE uses a sparse GRM format with three files:
//! - .sparseGRM.mtx: COO-format sparse matrix (row, col, value)
//! - .sparseGRM.mtx.sampleIDs.txt: Sample IDs (one per line)

use std::path::Path;

use anyhow::{Context, Result};

use saige_linalg::sparse::SparseMatrix;

/// Read a sparse GRM from file.
///
/// Returns (sparse_matrix, sample_ids).
pub fn read_sparse_grm(
    mtx_path: &Path,
    sample_ids_path: &Path,
) -> Result<(SparseMatrix, Vec<String>)> {
    // Read sample IDs
    let sample_ids: Vec<String> = std::fs::read_to_string(sample_ids_path)
        .with_context(|| format!("Failed to read sample IDs: {}", sample_ids_path.display()))?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let n = sample_ids.len();

    // Read MTX file
    let contents = std::fs::read_to_string(mtx_path)
        .with_context(|| format!("Failed to read sparse GRM: {}", mtx_path.display()))?;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut header_done = false;

    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with('%') || line.starts_with('#') {
            continue;
        }
        if !header_done {
            // First non-comment line is dimensions: nrows ncols nnz
            header_done = true;
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 3 {
            let row: usize = fields[0].parse::<usize>()? - 1; // 1-indexed to 0-indexed
            let col: usize = fields[1].parse::<usize>()? - 1;
            let val: f64 = fields[2].parse()?;
            rows.push(row);
            cols.push(col);
            vals.push(val);
            // Add symmetric entry if off-diagonal
            if row != col {
                rows.push(col);
                cols.push(row);
                vals.push(val);
            }
        }
    }

    let matrix = SparseMatrix::from_triplets(n, n, &rows, &cols, &vals);
    Ok((matrix, sample_ids))
}

/// Write a sparse GRM to file.
pub fn write_sparse_grm(
    matrix: &SparseMatrix,
    sample_ids: &[String],
    mtx_path: &Path,
    sample_ids_path: &Path,
) -> Result<()> {
    use std::io::Write;

    // Write sample IDs
    let mut f = std::fs::File::create(sample_ids_path)?;
    for id in sample_ids {
        writeln!(f, "{}", id)?;
    }

    // Write MTX file
    let mut f = std::fs::File::create(mtx_path)?;
    writeln!(f, "%%MatrixMarket matrix coordinate real symmetric")?;

    // Count upper-triangle entries (including diagonal)
    let n = matrix.nrows();
    let mut nnz = 0;
    let sprs_mat = matrix.as_sprs();
    let indptr = sprs_mat.indptr();
    let indices = sprs_mat.indices();
    let data = sprs_mat.data();

    let mut entries = Vec::new();
    for i in 0..n {
        let start = indptr.as_slice().unwrap()[i];
        let end = indptr.as_slice().unwrap()[i + 1];
        for idx in start..end {
            let j = indices[idx];
            if j >= i {
                entries.push((i, j, data[idx]));
                nnz += 1;
            }
        }
    }

    writeln!(f, "{} {} {}", n, n, nnz)?;
    for (i, j, v) in entries {
        writeln!(f, "{} {} {:.10}", i + 1, j + 1, v)?; // 1-indexed
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    #[test]
    fn test_read_write_sparse_grm() {
        let dir = tempfile::tempdir().unwrap();
        let mtx_path = dir.path().join("test.sparseGRM.mtx");
        let ids_path = dir.path().join("test.sparseGRM.mtx.sampleIDs.txt");

        // Create test files
        let mut f = std::fs::File::create(&mtx_path).unwrap();
        writeln!(f, "%%MatrixMarket matrix coordinate real symmetric").unwrap();
        writeln!(f, "3 3 4").unwrap();
        writeln!(f, "1 1 1.0").unwrap();
        writeln!(f, "2 2 1.0").unwrap();
        writeln!(f, "3 3 1.0").unwrap();
        writeln!(f, "1 2 0.25").unwrap();

        let mut f = std::fs::File::create(&ids_path).unwrap();
        writeln!(f, "S1").unwrap();
        writeln!(f, "S2").unwrap();
        writeln!(f, "S3").unwrap();

        let (matrix, ids) = read_sparse_grm(&mtx_path, &ids_path).unwrap();
        assert_eq!(ids, vec!["S1", "S2", "S3"]);
        assert_eq!(matrix.nrows(), 3);
        assert!((matrix.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((matrix.get(0, 1) - 0.25).abs() < 1e-10);
        assert!((matrix.get(1, 0) - 0.25).abs() < 1e-10); // symmetric
    }
}
