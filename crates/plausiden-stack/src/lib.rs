//! Heterogeneous compositional Stack architecture on HDC substrate.
//!
//! BUG ASSUMPTION: this crate is a v0 skeleton. The `Stack` type + operation
//! modes (`Dense`, `HrrBind`, `Identity`, `GatedRoute`, `Aggregate`) land in
//! the Phase 1+ task series. See `docs/ARCHITECTURE.md` (project root).
//!
//! Planned modules:
//!
//! - `stack`     — Stack + StackOfStacks types
//! - `op`        — Operation trait + impls (Dense / HrrBind / Identity / ...)
//! - `entropy`   — per-Stack tau parameter + stochasticity injection
//! - `meta`      — meta-controller for self-modification

#![forbid(unsafe_code)]

/// Returns a banner describing the crate state.
#[must_use]
pub fn banner() -> &'static str {
    concat!(
        "plausiden-stack v",
        env!("CARGO_PKG_VERSION"),
        " (Phase 0 scaffold — Stack APIs land in Phase 1)"
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
