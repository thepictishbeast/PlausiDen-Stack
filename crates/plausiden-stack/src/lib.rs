//! Heterogeneous compositional Stack architecture on HDC substrate.
//!
//! Phase 1 ships:
//!
//! - [`Operation`] enum with three modes: [`Operation::Identity`],
//!   [`Operation::Dense`] (HDC bind), [`Operation::HrrBind`] (FFT-based
//!   circular convolution).
//! - [`Stack`] struct: heterogeneous bundle of operations on a shared
//!   hypervector. `forward` applies every operation in parallel then
//!   bundles the outputs into one HDC vector. NOT MoE — single bundled
//!   output, not committee voting.
//!
//! Phase 2+ adds gated routing, stack-of-stacks recursion, entropy
//! gradients, and self-modification per `docs/ARCHITECTURE.md`.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]

pub mod gated;
pub mod op;
pub mod stack;

pub use gated::{GatedError, GatedStack};
pub use op::{Operation, OperationError};
pub use stack::{Stack, StackError};

/// Returns a banner describing the crate state.
#[must_use]
pub fn banner() -> &'static str {
    concat!(
        "plausiden-stack v",
        env!("CARGO_PKG_VERSION"),
        " (Phase 1 — Stack + Operation enum; Phase 2+ adds gating + recursion)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_mentions_version() {
        assert!(banner().contains(env!("CARGO_PKG_VERSION")));
    }
}
