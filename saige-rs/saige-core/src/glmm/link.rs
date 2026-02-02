//! Link functions for GLMMs.
//!
//! Maps between the linear predictor (eta) and the mean (mu).

/// Trait type for the GLMM.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TraitType {
    Binary,
    Quantitative,
    Survival,
}

/// Link function interface.
pub trait LinkFunction {
    /// Apply the link function: eta = g(mu).
    fn link(&self, mu: f64) -> f64;
    /// Apply the inverse link: mu = g^{-1}(eta).
    fn inv_link(&self, eta: f64) -> f64;
    /// Derivative of the inverse link: d(mu)/d(eta).
    fn inv_link_deriv(&self, eta: f64) -> f64;
    /// Variance function: V(mu).
    fn variance(&self, mu: f64) -> f64;
}

/// Logit link for binary traits.
#[derive(Debug, Clone, Copy)]
pub struct LogitLink;

impl LinkFunction for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        (mu / (1.0 - mu)).ln()
    }

    fn inv_link(&self, eta: f64) -> f64 {
        1.0 / (1.0 + (-eta).exp())
    }

    fn inv_link_deriv(&self, eta: f64) -> f64 {
        let p = self.inv_link(eta);
        p * (1.0 - p)
    }

    fn variance(&self, mu: f64) -> f64 {
        mu * (1.0 - mu)
    }
}

/// Identity link for quantitative traits.
#[derive(Debug, Clone, Copy)]
pub struct IdentityLink;

impl LinkFunction for IdentityLink {
    fn link(&self, mu: f64) -> f64 {
        mu
    }

    fn inv_link(&self, eta: f64) -> f64 {
        eta
    }

    fn inv_link_deriv(&self, _eta: f64) -> f64 {
        1.0
    }

    fn variance(&self, _mu: f64) -> f64 {
        1.0
    }
}

/// Log link for survival traits (complementary log-log).
#[derive(Debug, Clone, Copy)]
pub struct CloglogLink;

impl LinkFunction for CloglogLink {
    fn link(&self, mu: f64) -> f64 {
        (-(1.0 - mu).ln()).ln()
    }

    fn inv_link(&self, eta: f64) -> f64 {
        1.0 - (-eta.exp()).exp()
    }

    fn inv_link_deriv(&self, eta: f64) -> f64 {
        let e = eta.exp();
        e * (-e).exp()
    }

    fn variance(&self, mu: f64) -> f64 {
        mu * (1.0 - mu)
    }
}

/// Get the appropriate link function for a trait type.
pub fn get_link(trait_type: TraitType) -> Box<dyn LinkFunction + Send + Sync> {
    match trait_type {
        TraitType::Binary => Box::new(LogitLink),
        TraitType::Quantitative => Box::new(IdentityLink),
        TraitType::Survival => Box::new(CloglogLink),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logit() {
        let link = LogitLink;
        assert!((link.inv_link(0.0) - 0.5).abs() < 1e-10);
        assert!((link.link(0.5) - 0.0).abs() < 1e-10);
        // Round trip
        assert!((link.inv_link(link.link(0.3)) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_identity() {
        let link = IdentityLink;
        assert_eq!(link.link(5.0), 5.0);
        assert_eq!(link.inv_link(5.0), 5.0);
        assert_eq!(link.inv_link_deriv(5.0), 1.0);
    }

    #[test]
    fn test_cloglog() {
        let link = CloglogLink;
        // Round trip
        for &mu in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let eta = link.link(mu);
            let mu_back = link.inv_link(eta);
            assert!(
                (mu_back - mu).abs() < 1e-10,
                "cloglog round-trip failed for mu={}: got {}",
                mu,
                mu_back
            );
        }
        // inv_link(0) should give 1 - exp(-1) ≈ 0.6321
        let mu_at_zero = link.inv_link(0.0);
        assert!(
            (mu_at_zero - (1.0 - (-1.0_f64).exp())).abs() < 1e-10,
            "cloglog inv_link(0) = {}",
            mu_at_zero
        );
        // Variance: mu*(1-mu)
        assert!((link.variance(0.5) - 0.25).abs() < 1e-10);
        // Derivative should be positive for finite eta
        assert!(link.inv_link_deriv(0.0) > 0.0);
        assert!(link.inv_link_deriv(-1.0) > 0.0);
    }

    #[test]
    fn test_get_link() {
        let _ = get_link(TraitType::Binary);
        let _ = get_link(TraitType::Quantitative);
        let _ = get_link(TraitType::Survival);
    }
}
