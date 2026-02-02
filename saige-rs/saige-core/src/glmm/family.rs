//! GLMM family distributions.
//!
//! Defines the family (distribution + link) for the GLMM,
//! providing working weights and working residuals for IRLS.

use super::link::{LinkFunction, TraitType, get_link};

/// A GLMM family: distribution + link function.
pub struct Family {
    pub trait_type: TraitType,
    link: Box<dyn LinkFunction + Send + Sync>,
}

impl Family {
    pub fn new(trait_type: TraitType) -> Self {
        Self {
            trait_type,
            link: get_link(trait_type),
        }
    }

    /// Initialize mu from y.
    pub fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        match self.trait_type {
            TraitType::Binary => {
                // mu_init = (y + 0.5) / 2
                y.iter().map(|&yi| (yi + 0.5) / 2.0).collect()
            }
            TraitType::Quantitative => {
                // mu_init = y
                y.to_vec()
            }
            TraitType::Survival => {
                // mu_init = (y + 0.5) / 2, clipped to (0,1)
                y.iter()
                    .map(|&yi| ((yi + 0.5) / 2.0).clamp(0.01, 0.99))
                    .collect()
            }
        }
    }

    /// Compute linear predictor eta = g(mu).
    pub fn link(&self, mu: &[f64]) -> Vec<f64> {
        mu.iter().map(|&m| self.link.link(m)).collect()
    }

    /// Compute mu = g^{-1}(eta).
    pub fn inv_link(&self, eta: &[f64]) -> Vec<f64> {
        eta.iter().map(|&e| self.link.inv_link(e)).collect()
    }

    /// Compute working weights W = 1 / Var(Y_i) * (d(mu_i)/d(eta_i))^2.
    /// For binomial: W = mu * (1-mu)
    /// For Gaussian: W = 1
    pub fn working_weights(&self, mu: &[f64]) -> Vec<f64> {
        mu.iter()
            .map(|&m| {
                let v = self.link.variance(m);
                if v > 1e-30 { v } else { 1e-30 }
            })
            .collect()
    }

    /// Compute mu * (1 - mu) for binary traits, or 1 for quantitative.
    pub fn mu_eta2(&self, mu: &[f64]) -> Vec<f64> {
        match self.trait_type {
            TraitType::Binary | TraitType::Survival => {
                mu.iter().map(|&m| m * (1.0 - m)).collect()
            }
            TraitType::Quantitative => vec![1.0; mu.len()],
        }
    }

    /// Compute residuals: y - mu.
    pub fn residuals(&self, y: &[f64], mu: &[f64]) -> Vec<f64> {
        y.iter().zip(mu.iter()).map(|(&yi, &mi)| yi - mi).collect()
    }

    /// Update mu from eta, clamping to valid range.
    pub fn update_mu(&self, eta: &[f64]) -> Vec<f64> {
        let mu = self.inv_link(eta);
        match self.trait_type {
            TraitType::Binary | TraitType::Survival => {
                mu.iter()
                    .map(|&m| m.clamp(1e-10, 1.0 - 1e-10))
                    .collect()
            }
            TraitType::Quantitative => mu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_family() {
        let fam = Family::new(TraitType::Binary);
        let mu = fam.initialize_mu(&[0.0, 1.0, 1.0, 0.0]);
        assert!(mu.iter().all(|&m| m > 0.0 && m < 1.0));

        let w = fam.working_weights(&[0.5, 0.5]);
        assert!((w[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_quantitative_family() {
        let fam = Family::new(TraitType::Quantitative);
        let w = fam.working_weights(&[1.0, 2.0, 3.0]);
        assert!(w.iter().all(|&wi| (wi - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_survival_family() {
        let fam = Family::new(TraitType::Survival);

        // Initialize from survival outcomes (0/1 event indicators)
        let mu = fam.initialize_mu(&[0.0, 1.0, 1.0, 0.0]);
        // All mu should be in (0.01, 0.99)
        assert!(mu.iter().all(|&m| (0.01..=0.99).contains(&m)));

        // Working weights should be mu*(1-mu) (same form as binary)
        let w = fam.working_weights(&[0.5, 0.3]);
        assert!((w[0] - 0.25).abs() < 1e-10);
        assert!((w[1] - 0.21).abs() < 1e-10);

        // mu_eta2 should also be mu*(1-mu)
        let me2 = fam.mu_eta2(&[0.5, 0.3]);
        assert!((me2[0] - 0.25).abs() < 1e-10);

        // update_mu should clamp to (1e-10, 1-1e-10)
        let mu_updated = fam.update_mu(&[-10.0, 0.0, 10.0]);
        assert!(mu_updated[0] > 0.0);
        assert!(mu_updated[2] < 1.0);
        // Cloglog: inv_link(0) = 1 - exp(-1) ≈ 0.632
        assert!((mu_updated[1] - 0.6321205588).abs() < 1e-6);
    }
}
