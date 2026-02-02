//! Create a sparse GRM from genotype data.
//!
//! saige create-sparse-grm --plink-file ... --output-prefix ... --relatedness-cutoff 0.125

use anyhow::Result;
use clap::Args;
use tracing::info;

use saige_core::grm::sparse::compute_sparse_grm;
use saige_geno::plink::PlinkReader;
use saige_geno::sparse_grm_io::write_sparse_grm;
use saige_geno::traits::GenotypeReader;

#[derive(Args)]
pub struct CreateSparseGrmArgs {
    /// PLINK file prefix
    #[arg(long)]
    plink_file: String,

    /// Output file prefix
    #[arg(long)]
    output_prefix: String,

    /// Relatedness cutoff for sparse GRM
    #[arg(long, default_value = "0.125")]
    relatedness_cutoff: f64,

    /// Minimum MAF for markers
    #[arg(long, default_value = "0.01")]
    min_maf: f64,

    /// Number of samples to include (0 = all)
    #[arg(long, default_value = "0")]
    n_samples: usize,
}

pub fn run(args: CreateSparseGrmArgs) -> Result<()> {
    info!("=== Create Sparse GRM ===");
    info!("PLINK file: {}", args.plink_file);
    info!("Relatedness cutoff: {}", args.relatedness_cutoff);

    let mut reader = PlinkReader::new(&args.plink_file)?;
    info!(
        "Loaded {} markers x {} samples",
        reader.n_markers(),
        reader.n_samples()
    );

    let (sparse_grm, n_markers) =
        compute_sparse_grm(&mut reader, args.min_maf, args.relatedness_cutoff)?;

    let mtx_path = std::path::Path::new(&args.output_prefix).with_extension("sparseGRM.mtx");
    let ids_path =
        std::path::Path::new(&args.output_prefix).with_extension("sparseGRM.mtx.sampleIDs.txt");

    let sample_ids = reader.sample_ids().to_vec();
    write_sparse_grm(&sparse_grm, &sample_ids, &mtx_path, &ids_path)?;

    info!(
        "Sparse GRM saved: {} entries, {} markers used",
        sparse_grm.nnz(),
        n_markers
    );

    Ok(())
}
