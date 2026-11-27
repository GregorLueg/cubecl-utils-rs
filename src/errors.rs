//! Errors in `cubecl-utils-rs`.

use thiserror::Error;

use cubecl::server::ServerError;

/// All error variants that can occur across `cubecl-utils-rs` operations.
///
/// Marked `#[non_exhaustive]` so that added variants do not break a downstream
/// exhaustive `match`. Match on the variants you care about and keep a `_` arm.
///
/// Every variant here describes a condition that CubeCL itself reports either
/// silently or in the wrong place. See the crate-level docs for why that
/// matters.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CubeclUtilsErrors {
    /// A dispatch exceeds the device's per-dimension cube-count limit.
    ///
    /// Not a soft failure on wgpu: the launch is rejected on the CubeCL server
    /// thread, that thread dies, and every subsequent call on the client
    /// returns an unrelated `CallError` from somewhere else entirely.
    #[error(
        "Kernel '{kernel}' requested a cube count of {requested:?}, but this device's limit is \
         {limit:?}."
    )]
    CubeCountExceeded {
        /// Name of the kernel whose dispatch was rejected
        kernel: &'static str,
        /// Requested cube count as `(x, y, z)`
        requested: (u32, u32, u32),
        /// Per-dimension device limit as `(x, y, z)`
        limit: (u32, u32, u32),
    },

    /// A flat cube count cannot be expressed as a 2D grid within the limit.
    ///
    /// Reachable only when `total_cubes` exceeds `max_dim * max_dim`, i.e. past
    /// roughly 4.29e9 cubes against a 65535 limit.
    #[error(
        "A grid of {total_cubes} cubes does not fit a 2D decomposition bounded by {max_dim} per \
         dimension."
    )]
    GridTooLarge {
        /// Flat cube count that was requested
        total_cubes: u32,
        /// Per-dimension limit the decomposition had to respect
        max_dim: u32,
    },

    /// A single buffer exceeds the device's per-binding size limit.
    ///
    /// Over-sized bindings are rejected without an error surfacing: the kernel
    /// does no work and returns zeros, so the condition is caught on the host
    /// before the allocation instead.
    #[error(
        "A GPU buffer of {requested} bytes exceeds this device's per-binding limit of {limit} \
         bytes."
    )]
    BindingTooLarge {
        /// Bytes the buffer requires
        requested: u64,
        /// Per-binding limit reported by the device
        limit: u64,
    },

    /// A kernel's shared-memory footprint exceeds the device's per-workgroup
    /// budget.
    ///
    /// Carries the budget so the caller can report what would have fitted.
    /// Apple Silicon reports 32768 bytes, which is at the low end of what
    /// desktop hardware offers but above the 16384 that some integrated parts
    /// report.
    #[error(
        "Kernel '{kernel}' needs {requested} bytes of shared memory, but this device offers only \
         {available} bytes per workgroup."
    )]
    SharedMemoryExceeded {
        /// Name of the kernel whose staging does not fit
        kernel: &'static str,
        /// Bytes the kernel would allocate
        requested: usize,
        /// Bytes the device offers per workgroup
        available: usize,
    },

    /// A buffer holds fewer elements than a kernel will touch.
    ///
    /// Unchecked launches read and write past the end of a short buffer without
    /// an error, so the dispatchers check every operand on the host first.
    #[error(
        "Kernel '{kernel}' needs {required} elements in buffer '{buffer}', but it holds only \
         {provided}."
    )]
    BufferTooSmall {
        /// Name of the kernel that would have overrun the buffer
        kernel: &'static str,
        /// Name of the buffer that is too short
        buffer: &'static str,
        /// Elements the kernel touches
        required: usize,
        /// Elements the buffer holds
        provided: usize,
    },

    /// A GEMM's inner dimension exceeds what its shared-memory staging holds.
    #[error(
        "Kernel '{kernel}' stages an inner dimension of at most {max}, but {requested} was \
         requested."
    )]
    InnerDimTooLarge {
        /// Name of the kernel whose staging is too small
        kernel: &'static str,
        /// Inner dimension that was requested
        requested: usize,
        /// Largest inner dimension the kernel stages
        max: usize,
    },

    /// Propagate errors from the CubeCL runtime.
    #[error("Error from the cubecl runtime: {0}")]
    CubeClServerError(#[from] ServerError),
}
