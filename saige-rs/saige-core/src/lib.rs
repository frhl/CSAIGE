//! saige-core: Statistical algorithms for SAIGE-RS
//!
//! Implements all core statistical methods: GLMM fitting (AI-REML),
//! saddlepoint approximation, score tests, Firth's regression,
//! GRM construction, LD matrix computation, and model serialization.

pub mod firth;
pub mod glmm;
pub mod grm;
pub mod ld;
pub mod model;
pub mod score_test;
pub mod spa;
pub mod util;
