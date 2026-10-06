//! Accounting for one thread-owned device workspace. The tensor owner must
//! acknowledge allocation, submission and cleanup; this policy never calls a GPU.
use serde::Serialize;
use std::thread::{self, ThreadId};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    New,
    Reserved,
    Ready,
    Busy,
    Failed,
    Draining,
    Released,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Binding {
    pub device: u32,
    pub stream: u64,
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum WorkspaceError {
    #[error("workspace size must be nonzero and fit its byte budget")]
    Budget,
    #[error("workspace size or counter overflow")]
    Overflow,
    #[error("workspace belongs to another thread")]
    Owner,
    #[error("workspace is bound to a different device or stream")]
    Binding,
    #[error("invalid workspace transition from {0:?}")]
    State(State),
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Receipt {
    pub state: State,
    pub binding: Option<Binding>,
    pub required_bytes: u64,
    pub budget_bytes: u64,
    /// Reserved or potentially live storage, including uncertain failure paths.
    pub charged_bytes: u64,
    /// Storage explicitly acknowledged by the tensor owner as allocated.
    pub committed_bytes: u64,
    pub allocation_count: u32,
    /// Completed host submissions, not a claim of GPU completion.
    pub completed_calls: u64,
    pub release_count: u32,
    pub last_failure: Option<String>,
}

/// Deliberately not Clone: one lease accounts for one tensor lifetime.
#[derive(Debug)]
pub struct WorkspaceLease {
    owner: ThreadId,
    receipt: Receipt,
}

impl WorkspaceLease {
    pub fn new(
        shape: &[u64],
        element_bytes: u64,
        budget_bytes: u64,
    ) -> Result<Self, WorkspaceError> {
        if shape.is_empty() || shape.contains(&0) || element_bytes == 0 {
            return Err(WorkspaceError::Budget);
        }
        let required_bytes = shape.iter().try_fold(element_bytes, |bytes, &dimension| {
            bytes.checked_mul(dimension).ok_or(WorkspaceError::Overflow)
        })?;
        if required_bytes > budget_bytes {
            return Err(WorkspaceError::Budget);
        }
        Ok(Self {
            owner: thread::current().id(),
            receipt: Receipt {
                state: State::New,
                binding: None,
                required_bytes,
                budget_bytes,
                charged_bytes: 0,
                committed_bytes: 0,
                allocation_count: 0,
                completed_calls: 0,
                release_count: 0,
                last_failure: None,
            },
        })
    }

    fn owner(&self) -> Result<(), WorkspaceError> {
        if thread::current().id() != self.owner {
            return Err(WorkspaceError::Owner);
        }
        Ok(())
    }

    fn require(&self, state: State) -> Result<(), WorkspaceError> {
        self.owner()?;
        if self.receipt.state != state {
            return Err(WorkspaceError::State(self.receipt.state));
        }
        Ok(())
    }

    fn binding(&self, binding: Binding) -> Result<(), WorkspaceError> {
        if self
            .receipt
            .binding
            .is_some_and(|expected| expected != binding)
        {
            return Err(WorkspaceError::Binding);
        }
        Ok(())
    }

    pub fn reserve(&mut self, binding: Binding) -> Result<(), WorkspaceError> {
        self.require(State::New)?;
        self.binding(binding)?;
        self.receipt.binding = Some(binding);
        self.receipt.charged_bytes = self.receipt.required_bytes;
        self.receipt.state = State::Reserved;
        Ok(())
    }

    /// Only acknowledge after the normally initialized tensor was allocated.
    pub fn allocation_committed(&mut self) -> Result<(), WorkspaceError> {
        self.require(State::Reserved)?;
        self.receipt.committed_bytes = self.receipt.required_bytes;
        self.receipt.allocation_count = 1;
        self.receipt.state = State::Ready;
        Ok(())
    }

    /// Only for a known failure before a tensor exists. Uncertain allocation
    /// outcomes must use fail(), retaining the charge until explicit cleanup.
    pub fn rollback_reservation(&mut self) -> Result<(), WorkspaceError> {
        self.require(State::Reserved)?;
        self.receipt.charged_bytes = 0;
        self.receipt.state = State::New;
        Ok(())
    }

    pub fn begin_use(&mut self, binding: Binding) -> Result<(), WorkspaceError> {
        self.require(State::Ready)?;
        self.binding(binding)?;
        self.receipt.state = State::Busy;
        Ok(())
    }

    /// The caller has queued all reads on the bound stream, so the next use
    /// may overwrite the storage on that same stream. No device drain implied.
    pub fn finish_use(&mut self) -> Result<(), WorkspaceError> {
        self.require(State::Busy)?;
        let calls = self
            .receipt
            .completed_calls
            .checked_add(1)
            .ok_or(WorkspaceError::Overflow)?;
        self.receipt.completed_calls = calls;
        self.receipt.state = State::Ready;
        Ok(())
    }

    pub fn fail(&mut self, reason: String) -> Result<(), WorkspaceError> {
        self.owner()?;
        if self.receipt.state == State::Released {
            return Err(WorkspaceError::State(State::Released));
        }
        self.receipt.state = State::Failed;
        self.receipt.last_failure = Some(reason);
        Ok(())
    }

    pub fn begin_close(&mut self) -> Result<(), WorkspaceError> {
        self.owner()?;
        if matches!(
            self.receipt.state,
            State::Busy | State::Draining | State::Released
        ) {
            return Err(WorkspaceError::State(self.receipt.state));
        }
        self.receipt.state = State::Draining;
        Ok(())
    }

    /// Only acknowledge AFTER successful synchronization and tensor release.
    /// A failed synchronization uses fail() and leaves the charge intact.
    pub fn release_after_drain(&mut self) -> Result<(), WorkspaceError> {
        self.require(State::Draining)?;
        self.receipt.charged_bytes = 0;
        self.receipt.committed_bytes = 0;
        self.receipt.release_count = 1;
        self.receipt.state = State::Released;
        Ok(())
    }

    pub fn receipt(&self) -> &Receipt {
        &self.receipt
    }
}
