//! Core traits for genotype reading.

use anyhow::Result;

/// Information about a genetic marker (variant).
#[derive(Debug, Clone)]
pub struct MarkerInfo {
    /// Chromosome (e.g. "1", "22", "X").
    pub chrom: String,
    /// Position in base pairs.
    pub pos: u64,
    /// Marker/variant ID (e.g. rsID).
    pub id: String,
    /// Reference allele.
    pub ref_allele: String,
    /// Alternative allele.
    pub alt_allele: String,
}

/// Data for a single marker across all samples.
#[derive(Debug, Clone)]
pub struct MarkerData {
    /// Marker metadata.
    pub info: MarkerInfo,
    /// Dosage values for each sample (0.0 to 2.0).
    /// Missing values represented as NaN.
    pub dosages: Vec<f64>,
    /// Allele frequency of the alt allele.
    pub af: f64,
    /// Minor allele count.
    pub mac: f64,
    /// Number of non-missing samples.
    pub n_valid: usize,
    /// Whether the marker is imputed (true) or genotyped (false).
    pub is_imputed: bool,
    /// Imputation quality (info score), if available.
    pub info_score: Option<f64>,
}

impl MarkerData {
    /// Compute allele frequency from dosages.
    pub fn compute_af(dosages: &[f64]) -> (f64, f64, usize) {
        let mut sum = 0.0;
        let mut n = 0usize;
        for &d in dosages {
            if !d.is_nan() {
                sum += d;
                n += 1;
            }
        }
        let af = if n > 0 { sum / (2.0 * n as f64) } else { 0.0 };
        let mac = sum.min(2.0 * n as f64 - sum);
        (af, mac, n)
    }

    /// Impute missing dosages with twice the allele frequency.
    pub fn impute_missing(&mut self) {
        let impute_val = 2.0 * self.af;
        for d in &mut self.dosages {
            if d.is_nan() {
                *d = impute_val;
            }
        }
    }
}

/// Trait for reading genotype data from various file formats.
///
/// Each format (PLINK, BGEN, VCF, SAV, PGEN) implements this trait.
/// Static dispatch via generics in hot loops; dynamic dispatch
/// (`Box<dyn GenotypeReader>`) at the CLI level.
pub trait GenotypeReader: Send {
    /// Total number of markers in the file.
    fn n_markers(&self) -> usize;

    /// Total number of samples in the file.
    fn n_samples(&self) -> usize;

    /// Get the list of sample IDs.
    fn sample_ids(&self) -> &[String];

    /// Set a sample subset for reading. Only these samples will be
    /// included in subsequent `read_marker` calls.
    fn set_sample_subset(&mut self, ids: &[String]) -> Result<()>;

    /// Read genotype data for marker at the given index.
    fn read_marker(&mut self, index: u64) -> Result<MarkerData>;

    /// Get marker info without reading genotype data.
    fn marker_info(&self, index: u64) -> Result<MarkerInfo>;
}
