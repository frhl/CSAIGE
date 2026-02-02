//! Firth's penalized logistic regression.
//!
//! Implements Firth's bias-reduced logistic regression as a fallback
//! for variants where the standard score test or SPA may be unreliable
//! (typically rare variants with extreme case/control imbalance).

pub mod logistic;
