//! Genotype imputation and quality control utilities.

/// Impute missing genotypes with the mean (2 * allele frequency).
pub fn impute_missing_mean(dosages: &mut [f64], af: f64) {
    let impute_val = 2.0 * af;
    for d in dosages.iter_mut() {
        if d.is_nan() {
            *d = impute_val;
        }
    }
}

/// Center genotypes: subtract mean.
pub fn center_genotypes(dosages: &mut [f64]) {
    let (sum, n) = dosages.iter().fold(
        (0.0, 0),
        |(s, n), &d| {
            if !d.is_nan() {
                (s + d, n + 1)
            } else {
                (s, n)
            }
        },
    );
    let mean = if n > 0 { sum / n as f64 } else { 0.0 };
    for d in dosages.iter_mut() {
        if !d.is_nan() {
            *d -= mean;
        }
    }
}

/// Standardize genotypes: subtract mean and divide by std dev.
pub fn standardize_genotypes(dosages: &mut [f64]) {
    let (sum, sum_sq, n) = dosages.iter().fold((0.0, 0.0, 0), |(s, ss, n), &d| {
        if !d.is_nan() {
            (s + d, ss + d * d, n + 1)
        } else {
            (s, ss, n)
        }
    });
    if n == 0 {
        return;
    }
    let mean = sum / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    let sd = var.max(0.0).sqrt();

    if sd > 1e-10 {
        for d in dosages.iter_mut() {
            if !d.is_nan() {
                *d = (*d - mean) / sd;
            }
        }
    } else {
        for d in dosages.iter_mut() {
            *d = 0.0;
        }
    }
}

/// Flip alleles if the allele frequency > 0.5 (ensure minor allele).
pub fn flip_to_minor_allele(dosages: &mut [f64], af: &mut f64) -> bool {
    if *af > 0.5 {
        for d in dosages.iter_mut() {
            if !d.is_nan() {
                *d = 2.0 - *d;
            }
        }
        *af = 1.0 - *af;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impute_missing() {
        let mut g = vec![0.0, f64::NAN, 2.0, f64::NAN, 1.0];
        impute_missing_mean(&mut g, 0.3);
        assert_eq!(g[1], 0.6);
        assert_eq!(g[3], 0.6);
        assert_eq!(g[0], 0.0);
    }

    #[test]
    fn test_center() {
        let mut g = vec![0.0, 1.0, 2.0];
        center_genotypes(&mut g);
        assert!((g[0] - (-1.0)).abs() < 1e-10);
        assert!((g[1] - 0.0).abs() < 1e-10);
        assert!((g[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_flip() {
        let mut g = vec![0.0, 1.0, 2.0];
        let mut af = 0.7;
        let flipped = flip_to_minor_allele(&mut g, &mut af);
        assert!(flipped);
        assert_eq!(g[0], 2.0);
        assert_eq!(g[2], 0.0);
        assert!((af - 0.3).abs() < 1e-10);
    }
}
