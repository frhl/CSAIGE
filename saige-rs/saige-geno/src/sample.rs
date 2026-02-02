//! Sample ID intersection and matching logic.
//!
//! Handles the common operation of intersecting sample IDs across
//! genotype files, phenotype files, and GRM files.

use std::collections::HashMap;

/// Result of intersecting sample IDs from multiple sources.
#[derive(Debug, Clone)]
pub struct SampleIntersection {
    /// Sample IDs in the intersection, in the order they appear in the primary source.
    pub ids: Vec<String>,
    /// Indices into each source for the intersected samples.
    pub indices: Vec<Vec<usize>>,
}

/// Intersect sample IDs from multiple sources.
///
/// Returns the intersection in the order they appear in the first (primary) source.
/// `sources` is a list of sample ID vectors.
pub fn intersect_samples(sources: &[&[String]]) -> SampleIntersection {
    if sources.is_empty() {
        return SampleIntersection {
            ids: Vec::new(),
            indices: Vec::new(),
        };
    }

    // Build lookup maps for all sources
    let maps: Vec<HashMap<&str, usize>> = sources
        .iter()
        .map(|ids| {
            ids.iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i))
                .collect()
        })
        .collect();

    // Find intersection using the first source as primary
    let mut result_ids = Vec::new();
    let mut result_indices: Vec<Vec<usize>> = vec![Vec::new(); sources.len()];

    for (primary_idx, id) in sources[0].iter().enumerate() {
        let in_all = maps[1..].iter().all(|m| m.contains_key(id.as_str()));
        if in_all {
            result_ids.push(id.clone());
            result_indices[0].push(primary_idx);
            for (src, map) in maps[1..].iter().enumerate() {
                result_indices[src + 1].push(map[id.as_str()]);
            }
        }
    }

    SampleIntersection {
        ids: result_ids,
        indices: result_indices,
    }
}

/// Reorder a vector according to the given index mapping.
pub fn reorder_vec<T: Clone>(data: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| data[i].clone()).collect()
}

/// Reorder f64 vector (efficient, no Clone needed).
pub fn reorder_f64(data: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| data[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect_two_sources() {
        let src1: Vec<String> = vec!["A", "B", "C", "D"]
            .into_iter()
            .map(String::from)
            .collect();
        let src2: Vec<String> = vec!["C", "A", "E"].into_iter().map(String::from).collect();

        let result = intersect_samples(&[&src1, &src2]);
        assert_eq!(result.ids, vec!["A", "C"]);
        assert_eq!(result.indices[0], vec![0, 2]); // indices into src1
        assert_eq!(result.indices[1], vec![1, 0]); // indices into src2
    }

    #[test]
    fn test_intersect_three_sources() {
        let s1: Vec<String> = vec!["A", "B", "C"].into_iter().map(String::from).collect();
        let s2: Vec<String> = vec!["B", "C", "D"].into_iter().map(String::from).collect();
        let s3: Vec<String> = vec!["C", "B", "E"].into_iter().map(String::from).collect();

        let result = intersect_samples(&[&s1, &s2, &s3]);
        assert_eq!(result.ids, vec!["B", "C"]);
    }

    #[test]
    fn test_empty_intersection() {
        let s1: Vec<String> = vec!["A", "B"].into_iter().map(String::from).collect();
        let s2: Vec<String> = vec!["C", "D"].into_iter().map(String::from).collect();

        let result = intersect_samples(&[&s1, &s2]);
        assert!(result.ids.is_empty());
    }

    #[test]
    fn test_reorder_vec() {
        let data = vec![10, 20, 30, 40, 50];
        let reordered = reorder_vec(&data, &[4, 2, 0]);
        assert_eq!(reordered, vec![50, 30, 10]);
    }
}
