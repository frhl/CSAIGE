//! Step 3: LD matrix computation.
//!
//! saige ld-matrix --bgen-file ... --group-file ... --output-prefix ...

use clap::Args;
use anyhow::Result;
use tracing::info;

use saige_core::ld::matrix::{compute_ld_matrix, write_ld_matrix};
use saige_geno::group_file::GroupFile;
use saige_geno::traits::GenotypeReader;

#[derive(Args)]
pub struct LdMatrixArgs {
    /// BGEN file path
    #[arg(long)]
    bgen_file: Option<String>,

    /// PLINK file prefix
    #[arg(long)]
    plink_file: Option<String>,

    /// VCF file path
    #[arg(long)]
    vcf_file: Option<String>,

    /// Group file defining variant sets per gene
    #[arg(long)]
    group_file: String,

    /// Output file prefix
    #[arg(long)]
    output_prefix: String,

    /// Sample file for BGEN (if not embedded)
    #[arg(long)]
    sample_file: Option<String>,
}

pub fn run(args: LdMatrixArgs) -> Result<()> {
    info!("=== SAIGE Step 3: LD Matrix Computation ===");

    // Open genotype file
    let mut reader: Box<dyn GenotypeReader> = if let Some(ref bgen_path) = args.bgen_file {
        Box::new(saige_geno::bgen::BgenReader::new(bgen_path)?)
    } else if let Some(ref plink_path) = args.plink_file {
        Box::new(saige_geno::plink::PlinkReader::new(plink_path)?)
    } else if let Some(ref vcf_path) = args.vcf_file {
        Box::new(saige_geno::vcf::VcfReader::new(vcf_path)?)
    } else {
        anyhow::bail!("Must specify --bgen-file, --plink-file, or --vcf-file");
    };

    // Parse group file
    let group_file = GroupFile::parse(&args.group_file)?;
    let gene_names = group_file.gene_names();

    info!(
        "Computing LD matrices for {} genes, {} markers available",
        gene_names.len(),
        reader.n_markers()
    );

    // Build variant ID to index map
    let mut variant_id_map = std::collections::HashMap::new();
    for i in 0..reader.n_markers() {
        let info = reader.marker_info(i as u64)?;
        variant_id_map.insert(info.id.clone(), i);
    }

    for gene in &gene_names {
        let group = match group_file.group_for_gene(gene) {
            Some(g) => g,
            None => continue,
        };
        let all_variant_ids = group.variant_ids.clone();

        // Read dosages for these variants
        let mut dosages = Vec::new();
        let mut valid_ids = Vec::new();

        for vid in &all_variant_ids {
            if let Some(&idx) = variant_id_map.get(vid) {
                let data = reader.read_marker(idx as u64)?;
                dosages.push(data.dosages);
                valid_ids.push(vid.clone());
            }
        }

        if dosages.is_empty() {
            info!("Gene {}: no variants found, skipping", gene);
            continue;
        }

        // Compute LD matrix
        let ld = compute_ld_matrix(&dosages);

        // Write output
        let output_path = std::path::Path::new(&args.output_prefix)
            .join(format!("{}.ld.txt", gene));

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_ld_matrix(&ld, &valid_ids, &output_path)?;

        info!("Gene {}: {} variants, LD matrix written", gene, valid_ids.len());
    }

    info!("LD matrix computation complete");
    Ok(())
}
