#![allow(clippy::needless_range_loop)]
//! Dense matrix operations backed by faer.
//!
//! Wraps faer's column-major Mat<f64> with convenience methods
//! for the operations SAIGE uses most: matrix-vector products,
//! element-wise operations, and column/row access.

use faer::Mat;

/// A dense matrix wrapper around faer's `Mat<f64>`.
///
/// Column-major layout matching the Armadillo conventions
/// used in the original C++ code.
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    inner: Mat<f64>,
}

impl DenseMatrix {
    /// Create a new dense matrix filled with zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            inner: Mat::zeros(nrows, ncols),
        }
    }

    /// Create a new dense matrix filled with a constant value.
    pub fn full(nrows: usize, ncols: usize, value: f64) -> Self {
        Self {
            inner: Mat::from_fn(nrows, ncols, |_, _| value),
        }
    }

    /// Create a dense matrix from a flat vec (column-major order).
    pub fn from_col_major(nrows: usize, ncols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        let inner = Mat::from_fn(nrows, ncols, |i, j| data[j * nrows + i]);
        Self { inner }
    }

    /// Create a dense matrix from a 2D slice (row-major input).
    pub fn from_row_major(nrows: usize, ncols: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        let inner = Mat::from_fn(nrows, ncols, |i, j| data[i * ncols + j]);
        Self { inner }
    }

    /// Create an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        let inner = Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        Self { inner }
    }

    /// Create a column vector from a slice.
    pub fn from_vec(data: &[f64]) -> Self {
        let n = data.len();
        let inner = Mat::from_fn(n, 1, |i, _| data[i]);
        Self { inner }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.inner.read(row, col)
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.inner.write(row, col, value);
    }

    /// Get a reference to the underlying faer matrix.
    pub fn as_faer(&self) -> &Mat<f64> {
        &self.inner
    }

    /// Get a mutable reference to the underlying faer matrix.
    pub fn as_faer_mut(&mut self) -> &mut Mat<f64> {
        &mut self.inner
    }

    /// Consume self and return the underlying faer matrix.
    pub fn into_faer(self) -> Mat<f64> {
        self.inner
    }

    /// Create from a faer matrix.
    pub fn from_faer(mat: Mat<f64>) -> Self {
        Self { inner: mat }
    }

    /// Matrix-vector product: self * v -> result vector.
    pub fn mat_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.ncols(), v.len());
        let n = self.nrows();
        let mut result = vec![0.0; n];
        for j in 0..self.ncols() {
            let vj = v[j];
            for i in 0..n {
                result[i] += self.inner.read(i, j) * vj;
            }
        }
        result
    }

    /// Matrix-matrix product: self * other.
    pub fn mat_mul(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.ncols(), other.nrows());
        let result = &self.inner * &other.inner;
        DenseMatrix { inner: result }
    }

    /// Transpose.
    pub fn transpose(&self) -> DenseMatrix {
        let inner = self.inner.transpose().to_owned();
        DenseMatrix { inner }
    }

    /// Extract column as a Vec<f64>.
    pub fn col(&self, j: usize) -> Vec<f64> {
        let n = self.nrows();
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(self.inner.read(i, j));
        }
        v
    }

    /// Extract row as a Vec<f64>.
    pub fn row(&self, i: usize) -> Vec<f64> {
        let m = self.ncols();
        let mut v = Vec::with_capacity(m);
        for j in 0..m {
            v.push(self.inner.read(i, j));
        }
        v
    }

    /// Set an entire column from a slice.
    pub fn set_col(&mut self, j: usize, data: &[f64]) {
        assert_eq!(data.len(), self.nrows());
        for i in 0..self.nrows() {
            self.inner.write(i, j, data[i]);
        }
    }

    /// Set an entire row from a slice.
    pub fn set_row(&mut self, i: usize, data: &[f64]) {
        assert_eq!(data.len(), self.ncols());
        for j in 0..self.ncols() {
            self.inner.write(i, j, data[j]);
        }
    }

    /// Element-wise addition: self + other.
    pub fn add(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.nrows(), other.nrows());
        assert_eq!(self.ncols(), other.ncols());
        let inner = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.inner.read(i, j) + other.inner.read(i, j)
        });
        DenseMatrix { inner }
    }

    /// Element-wise subtraction: self - other.
    pub fn sub(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.nrows(), other.nrows());
        assert_eq!(self.ncols(), other.ncols());
        let inner = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.inner.read(i, j) - other.inner.read(i, j)
        });
        DenseMatrix { inner }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> DenseMatrix {
        let inner = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.inner.read(i, j) * s
        });
        DenseMatrix { inner }
    }

    /// Diagonal of a square matrix.
    pub fn diag(&self) -> Vec<f64> {
        let n = self.nrows().min(self.ncols());
        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            d.push(self.inner.read(i, i));
        }
        d
    }

    /// Create a diagonal matrix from a vector.
    pub fn from_diag(diag: &[f64]) -> Self {
        let n = diag.len();
        let inner = Mat::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 });
        Self { inner }
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let v = self.inner.read(i, j);
                sum += v * v;
            }
        }
        sum.sqrt()
    }

    /// Dot product of two column vectors (nx1 matrices).
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute X' * diag(w) * X for design matrix X and weight vector w.
    /// Returns a p x p matrix where p = X.ncols().
    pub fn xtwx(&self, w: &[f64]) -> DenseMatrix {
        let n = self.nrows();
        let p = self.ncols();
        assert_eq!(w.len(), n);
        let mut result = DenseMatrix::zeros(p, p);
        for j in 0..p {
            for k in j..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += self.inner.read(i, j) * w[i] * self.inner.read(i, k);
                }
                result.set(j, k, s);
                if j != k {
                    result.set(k, j, s);
                }
            }
        }
        result
    }

    /// Compute X' * diag(w) * v for design matrix X, weight vector w, and vector v.
    /// Returns a vector of length p = X.ncols().
    pub fn xtwv(&self, w: &[f64], v: &[f64]) -> Vec<f64> {
        let n = self.nrows();
        let p = self.ncols();
        assert_eq!(w.len(), n);
        assert_eq!(v.len(), n);
        let mut result = vec![0.0; p];
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += self.inner.read(i, j) * w[i] * v[i];
            }
            result[j] = s;
        }
        result
    }

    /// Extract column data as a flat Vec in column-major order.
    pub fn to_col_major(&self) -> Vec<f64> {
        let mut data = Vec::with_capacity(self.nrows() * self.ncols());
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                data.push(self.inner.read(i, j));
            }
        }
        data
    }
}

impl std::fmt::Display for DenseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                if j > 0 {
                    write!(f, "\t")?;
                }
                write!(f, "{:.6}", self.inner.read(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = DenseMatrix::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn test_identity() {
        let m = DenseMatrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(2, 2), 1.0);
    }

    #[test]
    fn test_mat_vec() {
        let m = DenseMatrix::identity(3);
        let v = vec![1.0, 2.0, 3.0];
        let result = m.mat_vec(&v);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mat_mul() {
        let a = DenseMatrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DenseMatrix::from_row_major(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.mat_mul(&b);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert!((c.get(0, 0) - 58.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 64.0).abs() < 1e-10);
        assert!((c.get(1, 0) - 139.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose() {
        let a = DenseMatrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let at = a.transpose();
        assert_eq!(at.nrows(), 3);
        assert_eq!(at.ncols(), 2);
        assert_eq!(at.get(0, 0), 1.0);
        assert_eq!(at.get(1, 0), 2.0);
        assert_eq!(at.get(0, 1), 4.0);
    }

    #[test]
    fn test_xtwx() {
        let x = DenseMatrix::from_row_major(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let w = vec![1.0, 2.0, 3.0];
        let result = x.xtwx(&w);
        // X'WX where W = diag(1,2,3)
        // col0: [1,0,1], col1: [0,1,1]
        // (0,0): 1*1*1 + 0*2*0 + 1*3*1 = 4
        // (0,1): 1*1*0 + 0*2*1 + 1*3*1 = 3
        // (1,1): 0*1*0 + 1*2*1 + 1*3*1 = 5
        assert!((result.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((result.get(0, 1) - 3.0).abs() < 1e-10);
        assert!((result.get(1, 0) - 3.0).abs() < 1e-10);
        assert!((result.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((DenseMatrix::dot(&a, &b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_diag() {
        let d = DenseMatrix::from_diag(&[2.0, 3.0, 5.0]);
        assert_eq!(d.get(0, 0), 2.0);
        assert_eq!(d.get(1, 1), 3.0);
        assert_eq!(d.get(2, 2), 5.0);
        assert_eq!(d.get(0, 1), 0.0);
    }
}
