//! Generalized Linear Mixed Model (GLMM) fitting.
//!
//! Implements the null model fitting procedure (Step 1):
//! - AI-REML for variance component estimation
//! - PCG solver for mixed model equations
//! - Variance ratio estimation
//! - LOCO (Leave-One-Chromosome-Out) procedure

pub mod ai_reml;
pub mod family;
pub mod link;
pub mod loco;
pub mod pcg;
pub mod variance_ratio;
