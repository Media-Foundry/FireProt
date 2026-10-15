//! Deterministic, bounded pLM batching policy; no tensors or GPU initialization.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub items: usize,
    pub max_batch_size: usize,
    /// Includes BOS/EOS and padding for every sequence in a microbatch.
    pub max_padded_tokens: usize,
    /// Exact lengths avoid padding; sorted lengths permit explicit padding.
    pub grouping: Grouping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grouping {
    ExactLength,
    LengthSorted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sequence {
    pub id: String,
    pub sequence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub schema_version: u32,
    pub sequences: Vec<Sequence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub sequences: Vec<Sequence>,
    /// Original input indices, in execution order. Outputs retain input order.
    pub groups: Vec<Vec<usize>>,
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=256).contains(&self.items)
            || !(1..=16).contains(&self.max_batch_size)
            || !(3..=16032).contains(&self.max_padded_tokens)
        {
            return Err(
                "pLM batch limits: items 1..256, size 1..16, padded tokens 3..16032".into(),
            );
        }
        Ok(())
    }
}

pub fn plan(bytes: &[u8], options: &Options) -> Result<Plan, String> {
    options.validate()?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("pLM batch input exceeds 16 MiB".into());
    }
    let input: Input = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if input.schema_version != 1 || input.sequences.len() != options.items {
        return Err("pLM batch input schema or item count mismatch".into());
    }
    let mut ids = BTreeSet::new();
    for item in &input.sequences {
        if item.id.is_empty()
            || item.id.len() > 128
            || !item
                .id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
            || !ids.insert(&item.id)
            || !(1..=1000).contains(&item.sequence.len())
            || !item
                .sequence
                .bytes()
                .all(|c| b"ACDEFGHIKLMNPQRSTVWYX".contains(&c))
        {
            return Err("invalid or duplicate pLM batch ID/sequence".into());
        }
        if item.sequence.len() + 2 > options.max_padded_tokens {
            return Err("one sequence exceeds the padded token budget".into());
        }
    }
    // Include the original index as a tie breaker, independent of sort stability.
    let mut order: Vec<usize> = (0..input.sequences.len()).collect();
    order.sort_by_key(|&i| (input.sequences[i].sequence.len(), i));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for index in order {
        let tokens = input.sequences[index].sequence.len() + 2;
        let append = groups.last().is_some_and(|group| {
            group.len() < options.max_batch_size
                && (group.len() + 1) * tokens <= options.max_padded_tokens
                && (options.grouping == Grouping::LengthSorted
                    || input.sequences[group[0]].sequence.len() + 2 == tokens)
        });
        if append {
            groups.last_mut().unwrap().push(index);
        } else {
            groups.push(vec![index]);
        }
    }
    Ok(Plan {
        sequences: input.sequences,
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> Options {
        Options {
            items: 4,
            max_batch_size: 4,
            max_padded_tokens: 4008,
            grouping: Grouping::ExactLength,
        }
    }
    fn input() -> Input {
        Input {
            schema_version: 1,
            sequences: [1000, 600, 1000, 800]
                .into_iter()
                .enumerate()
                .map(|(i, n)| Sequence {
                    id: format!("item-{i}"),
                    sequence: "A".repeat(n),
                })
                .collect(),
        }
    }
    #[test]
    fn groups_lengths_and_preserves_original_identity() {
        let input = input();
        let bytes = serde_json::to_vec(&input).unwrap();
        let mut opts = options();
        let result = plan(&bytes, &opts).unwrap();
        assert_eq!(result.groups, vec![vec![1], vec![3], vec![0, 2]]);
        assert_eq!(result.sequences, input.sequences);
        opts.grouping = Grouping::LengthSorted;
        assert_eq!(plan(&bytes, &opts).unwrap().groups, vec![vec![1, 3, 0, 2]]);
        opts.max_padded_tokens = 2004;
        assert_eq!(
            plan(&bytes, &opts).unwrap().groups,
            vec![vec![1, 3], vec![0, 2]]
        );
        opts.max_batch_size = 1;
        assert_eq!(
            plan(&bytes, &opts).unwrap().groups,
            vec![vec![1], vec![3], vec![0], vec![2]]
        );
    }
    #[test]
    fn rejects_ambiguous_or_unbounded_inputs() {
        let mut data = input();
        let mut opts = options();
        data.sequences[1].id = data.sequences[0].id.clone();
        assert!(plan(&serde_json::to_vec(&data).unwrap(), &opts).is_err());
        data = input();
        data.sequences[0].sequence = "A?".into();
        assert!(plan(&serde_json::to_vec(&data).unwrap(), &opts).is_err());
        opts.max_padded_tokens = 1001;
        assert!(plan(&serde_json::to_vec(&input()).unwrap(), &opts).is_err());
        opts = options();
        opts.items = 3;
        assert!(plan(&serde_json::to_vec(&input()).unwrap(), &opts).is_err());
        opts = options();
        opts.max_batch_size = 0;
        assert!(opts.validate().is_err());
    }
}
