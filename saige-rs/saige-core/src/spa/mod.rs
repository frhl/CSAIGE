//! Saddlepoint Approximation (SPA) for p-value computation.
//!
//! SPA provides more accurate p-values than the normal approximation
//! for the score test, especially in the tails of the distribution
//! when the trait is binary or survival.

pub mod binary;
pub mod survival;
pub mod fast;
