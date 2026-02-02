//! saige-geno: Genotype I/O abstraction for SAIGE-RS
//!
//! Provides a unified GenotypeReader trait and implementations for
//! PLINK bed/bim/fam, BGEN v1.2, VCF/BCF, SAV, and PGEN formats.

pub mod bgen;
pub mod group_file;
pub mod pgen;
pub mod phenotype;
pub mod plink;
pub mod sample;
pub mod sav;
pub mod sparse_grm_io;
pub mod traits;
pub mod vcf;

pub use traits::{GenotypeReader, MarkerData, MarkerInfo};
