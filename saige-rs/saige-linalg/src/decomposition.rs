#![allow(clippy::needless_range_loop)]
//! Matrix decompositions and solvers.
//!
//! Wrappers around faer for Cholesky, QR, eigendecomposition, and
//! the Preconditioned Conjugate Gradient (PCG) solver used in SAIGE
//! for solving (tau_e * W + tau_g * GRM) x = b.

use crate::dense::DenseMatrix;
use crate::sparse::SparseMatrix;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LinalgError {
    #[error("PCG failed to converge after {max_iter} iterations (residual: {residual:.2e})")]
    PcgNotConverged { max_iter: usize, residual: f64 },

    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,

    #[error("Singular matrix encountered")]
    SingularMatrix,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Result of a Cholesky decomposition.
pub struct CholeskyDecomp {
    /// Lower triangular factor L such that A = L * L'.
    pub l: DenseMatrix,
}

impl CholeskyDecomp {
    /// Compute the Cholesky decomposition of a symmetric positive definite matrix.
    pub fn new(a: &DenseMatrix) -> Result<Self, LinalgError> {
        let n = a.nrows();
        assert_eq!(n, a.ncols());
        let mut l = DenseMatrix::zeros(n, n);

        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l.get(j, k) * l.get(j, k);
            }
            let diag = a.get(j, j) - sum;
            if diag <= 0.0 {
                return Err(LinalgError::NotPositiveDefinite);
            }
            l.set(j, j, diag.sqrt());

            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                l.set(i, j, (a.get(i, j) - sum) / l.get(j, j));
            }
        }

        Ok(CholeskyDecomp { l })
    }

    /// Solve L * L' * x = b.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n);

        // Forward substitution: L * y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.l.get(i, j) * y[j];
            }
            y[i] = (b[i] - sum) / self.l.get(i, i);
        }

        // Backward substitution: L' * x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += self.l.get(j, i) * x[j];
            }
            x[i] = (y[i] - sum) / self.l.get(i, i);
        }

        x
    }

    /// Compute the inverse of the original matrix A^{-1} = (L L')^{-1}.
    pub fn inverse(&self) -> DenseMatrix {
        let n = self.l.nrows();
        let mut inv = DenseMatrix::zeros(n, n);
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = self.solve(&e);
            inv.set_col(j, &col);
        }
        inv
    }
}

/// Result of a QR decomposition: A = Q * R.
pub struct QrDecomp {
    pub q: DenseMatrix,
    pub r: DenseMatrix,
}

impl QrDecomp {
    /// Compute the thin QR decomposition of an m x n matrix (m >= n).
    /// Uses modified Gram-Schmidt.
    pub fn new(a: &DenseMatrix) -> Result<Self, LinalgError> {
        let m = a.nrows();
        let n = a.ncols();
        assert!(m >= n);

        let mut q = DenseMatrix::zeros(m, n);
        let mut r = DenseMatrix::zeros(n, n);

        // Copy columns of A
        let mut cols: Vec<Vec<f64>> = (0..n).map(|j| a.col(j)).collect();

        for j in 0..n {
            // Orthogonalize against previous columns
            for i in 0..j {
                let q_col = q.col(i);
                let rij = DenseMatrix::dot(&q_col, &cols[j]);
                r.set(i, j, rij);
                for k in 0..m {
                    cols[j][k] -= rij * q_col[k];
                }
            }

            let norm = DenseMatrix::dot(&cols[j], &cols[j]).sqrt();
            if norm < 1e-14 {
                return Err(LinalgError::SingularMatrix);
            }
            r.set(j, j, norm);
            for k in 0..m {
                q.set(k, j, cols[j][k] / norm);
            }
        }

        Ok(QrDecomp { q, r })
    }

    /// Solve R * x = Q' * b (least squares).
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.r.nrows();
        let qtb = self.q.transpose().mat_vec(b);

        // Back substitution: R * x = Q'b
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += self.r.get(i, j) * x[j];
            }
            x[i] = (qtb[i] - sum) / self.r.get(i, i);
        }
        x
    }
}

/// Preconditioned Conjugate Gradient (PCG) solver.
///
/// Solves A * x = b where A is symmetric positive definite.
/// Uses a preconditioner M^{-1} (provided as a function that applies M^{-1}).
///
/// In SAIGE, A = tau_e * W + tau_g * GRM, and M = diag(A).
pub struct PcgSolver {
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl Default for PcgSolver {
    fn default() -> Self {
        Self {
            tol: 1e-5,
            max_iter: 500,
        }
    }
}

/// Result of a PCG solve.
pub struct PcgResult {
    /// Solution vector x.
    pub x: Vec<f64>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual norm.
    pub residual: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

impl PcgSolver {
    pub fn new(tol: f64, max_iter: usize) -> Self {
        Self { tol, max_iter }
    }

    /// Solve A * x = b using PCG.
    ///
    /// - `mat_vec`: function computing A * v
    /// - `precond`: function computing M^{-1} * v (preconditioner)
    /// - `b`: right-hand side vector
    /// - `x0`: optional initial guess (if None, uses zero vector)
    pub fn solve<F, P>(&self, mat_vec: F, precond: P, b: &[f64], x0: Option<&[f64]>) -> PcgResult
    where
        F: Fn(&[f64]) -> Vec<f64>,
        P: Fn(&[f64]) -> Vec<f64>,
    {
        let n = b.len();

        // Initial guess
        let mut x: Vec<f64> = match x0 {
            Some(v) => v.to_vec(),
            None => vec![0.0; n],
        };

        // r = b - A*x
        let ax = mat_vec(&x);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();

        // z = M^{-1} * r
        let mut z = precond(&r);

        // p = z
        let mut p = z.clone();

        // rz = r' * z
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();

        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let tol_abs = self.tol * b_norm.max(1.0);

        for iter in 0..self.max_iter {
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol_abs {
                return PcgResult {
                    x,
                    iterations: iter,
                    residual: r_norm,
                    converged: true,
                };
            }

            // ap = A * p
            let ap = mat_vec(&p);

            // alpha = rz / (p' * ap)
            let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, ai)| pi * ai).sum();
            if pap.abs() < 1e-30 {
                return PcgResult {
                    x,
                    iterations: iter,
                    residual: r_norm,
                    converged: false,
                };
            }
            let alpha = rz / pap;

            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }

            // r = r - alpha * ap
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }

            // z = M^{-1} * r
            z = precond(&r);

            // beta = (r_new' * z_new) / (r_old' * z_old)
            let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
            let beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }

        let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        PcgResult {
            x,
            iterations: self.max_iter,
            residual: r_norm,
            converged: r_norm < tol_abs,
        }
    }

    /// Solve with a dense matrix A (for testing/small problems).
    pub fn solve_dense(&self, a: &DenseMatrix, b: &[f64]) -> PcgResult {
        let a_clone = a.clone();
        let diag = a.diag();

        let mat_vec = move |v: &[f64]| -> Vec<f64> { a_clone.mat_vec(v) };

        let precond = move |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(diag.iter())
                .map(|(vi, di)| if di.abs() > 1e-30 { vi / di } else { *vi })
                .collect()
        };

        self.solve(mat_vec, precond, b, None)
    }

    /// Solve with a sparse matrix A.
    pub fn solve_sparse(&self, a: &SparseMatrix, b: &[f64]) -> PcgResult {
        let a_clone = a.clone();
        let diag = a.diag();

        let mat_vec = move |v: &[f64]| -> Vec<f64> { a_clone.mat_vec(v) };

        let precond = move |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(diag.iter())
                .map(|(vi, di)| if di.abs() > 1e-30 { vi / di } else { *vi })
                .collect()
        };

        self.solve(mat_vec, precond, b, None)
    }
}

/// Compute eigenvalues of a symmetric matrix using the Jacobi method.
/// Returns eigenvalues sorted in descending order.
pub fn symmetric_eigenvalues(a: &DenseMatrix) -> Result<Vec<f64>, LinalgError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::DimensionMismatch {
            expected: n,
            got: a.ncols(),
        });
    }

    // Use faer for eigendecomposition
    let mat = a.as_faer();
    let eigenvalues = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eigenvalues.s();
    let mut evals: Vec<f64> = (0..n).map(|i| s.column_vector().read(i)).collect();
    evals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    Ok(evals)
}

/// Solve a symmetric positive definite system A*x = b using Cholesky.
pub fn solve_spd(a: &DenseMatrix, b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let chol = CholeskyDecomp::new(a)?;
    Ok(chol.solve(b))
}

/// Compute the inverse of a symmetric positive definite matrix.
pub fn inverse_spd(a: &DenseMatrix) -> Result<DenseMatrix, LinalgError> {
    let chol = CholeskyDecomp::new(a)?;
    Ok(chol.inverse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky() {
        // A = [[4, 2], [2, 3]]
        let a = DenseMatrix::from_row_major(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let chol = CholeskyDecomp::new(&a).unwrap();
        // L should be [[2, 0], [1, sqrt(2)]]
        assert!((chol.l.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((chol.l.get(1, 0) - 1.0).abs() < 1e-10);
        assert!((chol.l.get(1, 1) - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_solve() {
        let a = DenseMatrix::from_row_major(3, 3, &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);
        let b = vec![1.0, 2.0, 3.0];
        let chol = CholeskyDecomp::new(&a).unwrap();
        let x = chol.solve(&b);
        // Verify: A*x should equal b
        let ax = a.mat_vec(&x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "ax[{}]={} != b[{}]={}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn test_cholesky_not_pd() {
        // Not positive definite
        let a = DenseMatrix::from_row_major(2, 2, &[1.0, 3.0, 3.0, 1.0]);
        assert!(CholeskyDecomp::new(&a).is_err());
    }

    #[test]
    fn test_qr() {
        let a = DenseMatrix::from_row_major(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let qr = QrDecomp::new(&a).unwrap();
        // Q should be orthogonal: Q'Q = I
        let qtq = qr.q.transpose().mat_mul(&qr.q);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq.get(i, j) - expected).abs() < 1e-10,
                    "Q'Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq.get(i, j),
                    expected
                );
            }
        }
        // Q*R should equal A
        let qr_prod = qr.q.mat_mul(&qr.r);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (qr_prod.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "QR[{},{}] = {}, A[{},{}] = {}",
                    i,
                    j,
                    qr_prod.get(i, j),
                    i,
                    j,
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_qr_solve() {
        let a = DenseMatrix::from_row_major(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b = vec![1.0, 2.0, 2.0];
        let qr = QrDecomp::new(&a).unwrap();
        let x = qr.solve(&b);
        // Least squares solution: minimize ||Ax - b||
        // Check normal equations: A'Ax = A'b
        let ata = a.transpose().mat_mul(&a);
        let atb = a.transpose().mat_vec(&b);
        let atax = ata.mat_vec(&x);
        for i in 0..2 {
            assert!(
                (atax[i] - atb[i]).abs() < 1e-10,
                "A'Ax[{}]={} != A'b[{}]={}",
                i,
                atax[i],
                i,
                atb[i]
            );
        }
    }

    #[test]
    fn test_pcg_identity() {
        let a = DenseMatrix::identity(3);
        let b = vec![1.0, 2.0, 3.0];
        let pcg = PcgSolver::default();
        let result = pcg.solve_dense(&a, &b);
        assert!(result.converged);
        for i in 0..3 {
            assert!((result.x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pcg_spd() {
        let a = DenseMatrix::from_row_major(3, 3, &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);
        let b = vec![1.0, 2.0, 3.0];
        let pcg = PcgSolver::new(1e-10, 1000);
        let result = pcg.solve_dense(&a, &b);
        assert!(result.converged);
        // Verify A*x = b
        let ax = a.mat_vec(&result.x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "PCG: ax[{}]={} != b[{}]={}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn test_pcg_sparse() {
        let a = SparseMatrix::from_triplets(
            3,
            3,
            &[0, 0, 0, 1, 1, 1, 2, 2, 2],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
        );
        let b = vec![1.0, 2.0, 3.0];
        let pcg = PcgSolver::new(1e-10, 1000);
        let result = pcg.solve_sparse(&a, &b);
        assert!(result.converged);
        let ax = a.mat_vec(&result.x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "PCG sparse: ax[{}]={} != b[{}]={}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn test_eigenvalues() {
        // Symmetric matrix with known eigenvalues
        let a = DenseMatrix::from_row_major(2, 2, &[3.0, 1.0, 1.0, 3.0]);
        let evals = symmetric_eigenvalues(&a).unwrap();
        // Eigenvalues should be 4 and 2
        assert!((evals[0] - 4.0).abs() < 1e-10);
        assert!((evals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_spd() {
        let a = DenseMatrix::from_row_major(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let inv = inverse_spd(&a).unwrap();
        // A * A^{-1} should be I
        let prod = a.mat_mul(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.get(i, j) - expected).abs() < 1e-10,
                    "A*A^{{-1}}[{},{}] = {}, expected {}",
                    i,
                    j,
                    prod.get(i, j),
                    expected
                );
            }
        }
    }
}
