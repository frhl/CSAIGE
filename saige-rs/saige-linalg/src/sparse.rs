#![allow(clippy::needless_range_loop)]
//! Sparse matrix operations backed by sprs.
//!
//! Provides CSR/CSC sparse matrix types used for sparse GRM storage
//! and sparse linear algebra in SAIGE.

use sprs::{CsMat, CsMatI, TriMat};

/// A sparse matrix wrapper around sprs CSR format.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    inner: CsMatI<f64, usize>,
    nrows: usize,
    ncols: usize,
}

impl SparseMatrix {
    /// Create a sparse matrix from COO (coordinate) triplets.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> Self {
        assert_eq!(rows.len(), cols.len());
        assert_eq!(rows.len(), vals.len());
        let mut tri = TriMat::new((nrows, ncols));
        for i in 0..rows.len() {
            tri.add_triplet(rows[i], cols[i], vals[i]);
        }
        let csr = tri.to_csr();
        Self {
            inner: csr,
            nrows,
            ncols,
        }
    }

    /// Create from a dense matrix (keeps only non-zero entries).
    pub fn from_dense(data: &[f64], nrows: usize, ncols: usize) -> Self {
        let mut tri = TriMat::new((nrows, ncols));
        for j in 0..ncols {
            for i in 0..nrows {
                let val = data[j * nrows + i]; // column-major
                if val != 0.0 {
                    tri.add_triplet(i, j, val);
                }
            }
        }
        let csr = tri.to_csr();
        Self {
            inner: csr,
            nrows,
            ncols,
        }
    }

    /// Create a sparse identity matrix.
    pub fn identity(n: usize) -> Self {
        let inner = CsMat::eye(n);
        Self {
            inner,
            nrows: n,
            ncols: n,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Get element at (row, col). Returns 0.0 if not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        match self.inner.get(row, col) {
            Some(&v) => v,
            None => 0.0,
        }
    }

    /// Sparse matrix-vector product: self * v.
    pub fn mat_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.ncols);
        let mut result = vec![0.0; self.nrows];
        // CSR format: iterate over rows
        let indptr = self.inner.indptr();
        let indices = self.inner.indices();
        let data = self.inner.data();
        for i in 0..self.nrows {
            let start = indptr.as_slice().unwrap()[i];
            let end = indptr.as_slice().unwrap()[i + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += data[idx] * v[indices[idx]];
            }
            result[i] = sum;
        }
        result
    }

    /// Get a reference to the underlying sprs matrix.
    pub fn as_sprs(&self) -> &CsMatI<f64, usize> {
        &self.inner
    }

    /// Consume self and return the underlying sprs matrix.
    pub fn into_sprs(self) -> CsMatI<f64, usize> {
        self.inner
    }

    /// Create from a sprs CSR matrix.
    pub fn from_sprs(mat: CsMatI<f64, usize>) -> Self {
        let nrows = mat.rows();
        let ncols = mat.cols();
        Self {
            inner: mat,
            nrows,
            ncols,
        }
    }

    /// Extract the diagonal entries.
    pub fn diag(&self) -> Vec<f64> {
        let n = self.nrows.min(self.ncols);
        let mut d = vec![0.0; n];
        for i in 0..n {
            d[i] = self.get(i, i);
        }
        d
    }

    /// Scale all entries by a scalar.
    pub fn scale(&self, s: f64) -> SparseMatrix {
        let scaled = self.inner.map(|v| v * s);
        SparseMatrix {
            inner: scaled,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Add two sparse matrices (same dimensions).
    pub fn add(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        let result = &self.inner + &other.inner;
        SparseMatrix {
            inner: result,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Apply a relatedness cutoff: zero out entries below the threshold.
    /// Keeps only entries where |value| >= cutoff.
    pub fn threshold(&self, cutoff: f64) -> SparseMatrix {
        let mut tri = TriMat::new((self.nrows, self.ncols));
        let indptr = self.inner.indptr();
        let indices = self.inner.indices();
        let data = self.inner.data();
        for i in 0..self.nrows {
            let start = indptr.as_slice().unwrap()[i];
            let end = indptr.as_slice().unwrap()[i + 1];
            for idx in start..end {
                let val = data[idx];
                if val.abs() >= cutoff {
                    tri.add_triplet(i, indices[idx], val);
                }
            }
        }
        let csr = tri.to_csr();
        SparseMatrix {
            inner: csr,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let m = SparseMatrix::identity(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn test_mat_vec() {
        let m =
            SparseMatrix::from_triplets(3, 3, &[0, 1, 2, 0], &[0, 1, 2, 2], &[1.0, 2.0, 3.0, 0.5]);
        let v = vec![1.0, 1.0, 1.0];
        let result = m.mat_vec(&v);
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_triplets() {
        let m = SparseMatrix::from_triplets(2, 2, &[0, 1], &[0, 1], &[3.0, 7.0]);
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(1, 1), 7.0);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn test_threshold() {
        let m = SparseMatrix::from_triplets(
            3,
            3,
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 1, 2],
            &[1.0, 0.05, 0.05, 1.0, 0.2, 1.0],
        );
        let t = m.threshold(0.1);
        assert_eq!(t.get(0, 1), 0.0); // below threshold
        assert_eq!(t.get(2, 1), 0.2); // above threshold
    }

    #[test]
    fn test_diag() {
        let m = SparseMatrix::from_triplets(3, 3, &[0, 1, 2], &[0, 1, 2], &[2.0, 4.0, 6.0]);
        let d = m.diag();
        assert_eq!(d, vec![2.0, 4.0, 6.0]);
    }
}
