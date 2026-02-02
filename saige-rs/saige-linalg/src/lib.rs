//! saige-linalg: Linear algebra wrappers for SAIGE-RS
//!
//! Provides dense and sparse matrix operations, decompositions,
//! and the PCG (Preconditioned Conjugate Gradient) solver used
//! throughout SAIGE's statistical algorithms.

pub mod dense;
pub mod sparse;
pub mod decomposition;

pub use dense::DenseMatrix;
pub use sparse::SparseMatrix;
