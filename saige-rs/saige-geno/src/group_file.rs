//! Gene/region group file parser for region-based tests.
//!
//! Group files define sets of variants belonging to gene regions,
//! with optional annotation categories and weights.
//!
//! SAIGE group file format (paired lines per gene):
//! ```text
//! GENE1 var 1:1:A:C 1:2:A:C 1:3:A:C
//! GENE1 anno lof lof missense
//! GENE2 var 1:51:A:C 1:52:A:C
//! GENE2 anno missense lof
//! ```
//!
//! Optional third line for weights:
//! ```text
//! GENE1 var 1:1:A:C 1:2:A:C
//! GENE1 anno lof missense
//! GENE1 weight 0.5 0.3
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

/// A group of variants for region-based testing.
#[derive(Debug, Clone)]
pub struct VariantGroup {
    /// Gene or region name.
    pub name: String,
    /// Variant IDs in this group.
    pub variant_ids: Vec<String>,
    /// Per-variant annotation categories (e.g., "lof", "missense").
    pub annotations: Vec<String>,
    /// Optional weights for each variant.
    pub weights: Option<Vec<f64>>,
}

impl VariantGroup {
    /// Get unique annotation categories in this group.
    pub fn unique_annotations(&self) -> Vec<String> {
        let mut annos: Vec<String> = self
            .annotations
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        annos.sort();
        annos
    }

    /// Get indices of variants with a specific annotation.
    pub fn indices_for_annotation(&self, anno: &str) -> Vec<usize> {
        self.annotations
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                // Handle semicolon-separated annotations like "lof;missense"
                a.split(';').any(|part| part == anno)
            })
            .map(|(i, _)| i)
            .collect()
    }
}

/// All groups parsed from a group file.
#[derive(Debug, Clone)]
pub struct GroupFile {
    pub groups: Vec<VariantGroup>,
}

impl GroupFile {
    /// Parse a SAIGE group file.
    ///
    /// Format: paired lines per gene (var line + anno line, optional weight line).
    pub fn parse<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read group file: {}", path.display()))?;

        let lines: Vec<&str> = contents
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();

        // Parse into gene -> (variants, annotations, weights)
        type GeneEntry = (Vec<String>, Vec<String>, Option<Vec<f64>>);
        let mut gene_data: HashMap<String, GeneEntry> = HashMap::new();
        // Track insertion order
        let mut gene_order: Vec<String> = Vec::new();

        for line in &lines {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 3 {
                continue;
            }

            let gene = fields[0].to_string();
            let line_type = fields[1];

            if !gene_data.contains_key(&gene) {
                gene_data.insert(gene.clone(), (Vec::new(), Vec::new(), None));
                gene_order.push(gene.clone());
            }

            let entry = gene_data.get_mut(&gene).unwrap();

            match line_type {
                "var" => {
                    entry.0 = fields[2..].iter().map(|s| s.to_string()).collect();
                }
                "anno" => {
                    entry.1 = fields[2..].iter().map(|s| s.to_string()).collect();
                }
                "weight" => {
                    let weights: Vec<f64> = fields[2..]
                        .iter()
                        .map(|s| s.parse::<f64>().unwrap_or(1.0))
                        .collect();
                    entry.2 = Some(weights);
                }
                _ => {
                    // Unknown line type - try to parse as old format (no line type keyword)
                    // Old format: GENE var1 var2 var3
                    if entry.0.is_empty() {
                        entry.0 = fields[1..].iter().map(|s| s.to_string()).collect();
                    }
                }
            }
        }

        let mut groups = Vec::new();
        for gene in &gene_order {
            let (variants, annotations, weights) = gene_data.remove(gene).unwrap();
            if variants.is_empty() {
                continue;
            }

            // If no annotations provided, default to empty strings
            let annotations = if annotations.is_empty() {
                vec!["".to_string(); variants.len()]
            } else if annotations.len() != variants.len() {
                // Pad or truncate annotations to match variant count
                let mut a = annotations;
                a.resize(variants.len(), "".to_string());
                a
            } else {
                annotations
            };

            groups.push(VariantGroup {
                name: gene.clone(),
                variant_ids: variants,
                annotations,
                weights,
            });
        }

        Ok(GroupFile { groups })
    }

    /// Get all unique gene/region names.
    pub fn gene_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .groups
            .iter()
            .map(|g| g.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }

    /// Get the group for a specific gene.
    pub fn group_for_gene(&self, gene: &str) -> Option<&VariantGroup> {
        self.groups.iter().find(|g| g.name == gene)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_group_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("groups.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "GENE1 var 1:1:A:C 1:2:A:C 1:3:A:C").unwrap();
        writeln!(f, "GENE1 anno lof lof missense").unwrap();
        writeln!(f, "GENE2 var 1:4:A:C 1:5:A:C").unwrap();
        writeln!(f, "GENE2 anno lof lof").unwrap();

        let gf = GroupFile::parse(&path).unwrap();
        assert_eq!(gf.groups.len(), 2);
        assert_eq!(gf.groups[0].name, "GENE1");
        assert_eq!(
            gf.groups[0].variant_ids,
            vec!["1:1:A:C", "1:2:A:C", "1:3:A:C"]
        );
        assert_eq!(gf.groups[0].annotations, vec!["lof", "lof", "missense"]);
        assert_eq!(gf.groups[0].unique_annotations(), vec!["lof", "missense"]);
        assert_eq!(gf.groups[0].indices_for_annotation("lof"), vec![0, 1]);
        assert_eq!(gf.groups[0].indices_for_annotation("missense"), vec![2]);
        assert_eq!(gf.gene_names(), vec!["GENE1", "GENE2"]);
    }

    #[test]
    fn test_parse_group_file_with_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("groups.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "GENE1 var 1:1:A:C 1:2:A:C").unwrap();
        writeln!(f, "GENE1 anno lof missense").unwrap();
        writeln!(f, "GENE1 weight 0.5 0.3").unwrap();

        let gf = GroupFile::parse(&path).unwrap();
        assert_eq!(gf.groups.len(), 1);
        assert_eq!(gf.groups[0].weights, Some(vec![0.5, 0.3]));
    }

    #[test]
    fn test_semicolon_annotations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("groups.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "GENE1 var v1 v2 v3").unwrap();
        writeln!(f, "GENE1 anno lof lof;missense missense").unwrap();

        let gf = GroupFile::parse(&path).unwrap();
        // v2 has "lof;missense" so it should match both
        assert_eq!(gf.groups[0].indices_for_annotation("lof"), vec![0, 1]);
        assert_eq!(gf.groups[0].indices_for_annotation("missense"), vec![1, 2]);
    }
}
