//! Deterministic operator policy. GPU function invocation remains at the model
//! boundary; the registry returns an implementation identity, never a tensor.
use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationSignature {
    pub operation: String,
    pub device_type: String,
    pub architecture: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Exact strides when known; layout names alone do not identify BLAS layout.
    pub strides: Vec<i64>,
    pub layout: String,
    pub mode: String,
    pub mask: String,
    pub inplace: bool,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Implementation {
    pub operation: String,
    pub name: String,
    pub version: String,
    pub priority: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Rejection {
    pub implementation: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Selection {
    /// None always means the caller must use native execution.
    pub implementation: Option<Implementation>,
    pub rejected: Vec<Rejection>,
    pub cache_hit: bool,
}

type Capability = Box<dyn Fn(&OperationSignature) -> Result<(), String> + Send + Sync>;
struct Candidate {
    metadata: Implementation,
    supports: Capability,
}

/// Single-owner policy state. The in-process bridge supplies synchronization;
/// mutation requires exclusive access, so registration cannot race a resolution.
#[derive(Default)]
pub struct Registry {
    candidates: BTreeMap<String, Vec<Candidate>>,
    cache: BTreeMap<OperationSignature, Selection>,
    counts: BTreeMap<(String, String), u64>,
}

impl Registry {
    pub fn register(
        &mut self,
        metadata: Implementation,
        supports: impl Fn(&OperationSignature) -> Result<(), String> + Send + Sync + 'static,
    ) -> Result<(), RegistryError> {
        if [&metadata.operation, &metadata.name, &metadata.version]
            .iter()
            .any(|s| s.trim().is_empty())
        {
            return Err(RegistryError::EmptyIdentity);
        }
        let candidates = self
            .candidates
            .entry(metadata.operation.clone())
            .or_default();
        if candidates.iter().any(|c| c.metadata.name == metadata.name) {
            return Err(RegistryError::Duplicate);
        }
        candidates.push(Candidate {
            metadata,
            supports: Box::new(supports),
        });
        candidates.sort_by(|a, b| {
            b.metadata
                .priority
                .cmp(&a.metadata.priority)
                .then_with(|| a.metadata.name.cmp(&b.metadata.name))
                .then_with(|| a.metadata.version.cmp(&b.metadata.version))
        });
        self.cache.clear();
        Ok(())
    }

    pub fn unregister(&mut self, operation: &str, name: &str) -> Result<(), RegistryError> {
        let candidates = self
            .candidates
            .get_mut(operation)
            .ok_or(RegistryError::Unknown)?;
        let index = candidates
            .iter()
            .position(|c| c.metadata.name == name)
            .ok_or(RegistryError::Unknown)?;
        candidates.remove(index);
        self.cache.clear();
        Ok(())
    }

    pub fn resolve(&mut self, signature: &OperationSignature) -> Selection {
        let result = if let Some(cached) = self.cache.get(signature) {
            Selection {
                cache_hit: true,
                ..cached.clone()
            }
        } else {
            let mut result = Selection {
                implementation: None,
                rejected: Vec::new(),
                cache_hit: false,
            };
            for candidate in self
                .candidates
                .get(&signature.operation)
                .into_iter()
                .flatten()
            {
                // An implementation predicate is an extension boundary. A panic
                // rejects that candidate; it must never select unchecked code.
                match catch_unwind(AssertUnwindSafe(|| (candidate.supports)(signature))) {
                    Ok(Ok(())) => {
                        result.implementation = Some(candidate.metadata.clone());
                        break;
                    }
                    Ok(Err(reason)) => result.rejected.push(Rejection {
                        implementation: candidate.metadata.name.clone(),
                        reason,
                    }),
                    Err(_) => result.rejected.push(Rejection {
                        implementation: candidate.metadata.name.clone(),
                        reason: "capability predicate panicked".into(),
                    }),
                }
            }
            self.cache.insert(signature.clone(), result.clone());
            result
        };
        let name = result
            .implementation
            .as_ref()
            .map_or("native", |m| m.name.as_str());
        let count = self
            .counts
            .entry((signature.operation.clone(), name.to_owned()))
            .or_default();
        *count = count.saturating_add(1);
        result
    }

    pub fn count(&self, operation: &str, implementation: &str) -> u64 {
        self.counts
            .get(&(operation.into(), implementation.into()))
            .copied()
            .unwrap_or(0)
    }

    /// Capability policy changes outside registration require explicit cache
    /// invalidation before the next resolution, matching process-start policies.
    pub fn clear_runtime_state(&mut self) {
        self.cache.clear();
        self.counts.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryError {
    #[error("operation, implementation name and version must not be empty")]
    EmptyIdentity,
    #[error("implementation name is already registered for this operation")]
    Duplicate,
    #[error("implementation is not registered for this operation")]
    Unknown,
}
