//! Score tests for genetic association.
//!
//! Implements single-variant and region-based score tests with
//! variance ratio adjustment and SPA correction.

pub mod single_variant;
pub mod region;
pub mod cct;
pub mod exact;
pub mod permutation;
