//! Compute effective sample size from a fitted model.
//!
//! saige get-neff --model-file ...

use clap::Args;
use anyhow::Result;

use saige_core::model::serialization::load_model;

#[derive(Args)]
pub struct GetNeffArgs {
    /// Model file from Step 1 (.saige.model)
    #[arg(long)]
    model_file: String,
}

pub fn run(args: GetNeffArgs) -> Result<()> {
    let model = load_model(std::path::Path::new(&args.model_file))?;

    let n_eff = model.n_eff();

    println!("Model: {}", args.model_file);
    println!("Trait type: {:?}", model.trait_type);
    println!("N samples: {}", model.n_samples);
    println!("N_eff: {:.2}", n_eff);
    println!("Tau: [{:.6}, {:.6}]", model.tau[0], model.tau[1]);
    println!("Variance ratio: {:.4}", model.variance_ratio.variance_ratio);

    Ok(())
}
