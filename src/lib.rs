//! Shared CubeCL helpers: GPU tensors, device-limit queries and validated
//! dispatch geometry. No algorithms and no kernels.
//!
//! # Why this exists
//!
//! A CubeCL kernel dispatched with `launch_unchecked` that busts a device limit
//! does not fail loudly:
//!
//! - Over-allocating **shared memory** makes the kernel do no work. It writes
//!   nothing, returns zeros and reports no error. Downstream code then reads
//!   uninitialised memory, which surfaces as an absurd index or a distance in
//!   an index slot rather than as anything pointing at the kernel.
//! - Over-sizing a **binding** does the same.
//! - Busting the **cube-count** limit is worse: the launch is rejected on the
//!   CubeCL server thread, that thread dies, and the next unrelated call on the
//!   client returns a `CallError` from somewhere else entirely.
//!
//! So device limits are a correctness concern, not a tuning one, and they are
//! easy to get wrong when every machine to hand reports the same numbers. Apple
//! Silicon via wgpu reports 32 KiB of shared memory, 65535 cubes per grid
//! dimension and a plane size pinned to exactly 32. None of that is portable.
//!
//! # Design
//!
//! Every limit decision is a pure function of [`GpuLimits`]. Only
//! [`GpuLimits::from_client`] and the [`GpuTensor`] constructors touch a
//! `ComputeClient`; everything else takes limits as data.
//!
//! That is what makes the awkward cases testable. Asserting that a staging plan
//! shrinks correctly on a 16 KiB device, or that a workgroup rounds to whole
//! wave64 planes, needs no such device to be present.
//!
//! ```no_run
//! # use cubecl::prelude::*;
//! use cubecl_utils_rs::prelude::*;
//!
//! # fn demo<R: Runtime>(client: &ComputeClient<R>, n_blocks: u32) -> Result<(), CubeclUtilsErrors> {
//! let limits = GpuLimits::from_client(client);
//! let (gx, gy) = grid_2d(n_blocks, &limits)?;
//! let count = checked_cube_count("my_kernel", gx, gy, 1, &limits)?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod errors;
pub mod layout;
pub mod limits;
pub mod prelude;
pub mod tensor;
pub mod traits;

pub use crate::errors::CubeclUtilsErrors;
pub use crate::layout::{pad_vectors, padded_dim, LINE_SIZE};
pub use crate::limits::{
    checked_cube_count, fits_binding, fits_shared_memory, grid_2d, grid_2d_limited,
    plane_partitions, plane_uniform, resident_workgroups, resolve_workgroup_size, GpuLimits,
};
pub use crate::tensor::GpuTensor;
pub use crate::traits::CubeclFloat;
