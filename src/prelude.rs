//! User-facing re-exports.
//!
//! `use cubecl_utils_rs::prelude::*;` brings in everything needed to size a
//! dispatch, allocate a tensor and run a GEMM.

pub use crate::errors::CubeclUtilsErrors;
pub use crate::gemm::{
    skinny_chunks, skinny_gemm, skinny_partial_elems, tall_skinny_gemm, TSMM_K_MAX,
};
pub use crate::layout::{pad_vectors, padded_dim, LINE_SIZE};
pub use crate::limits::{
    checked_cube_count, fits_binding, fits_shared_memory, grid_2d, grid_2d_limited,
    plane_partitions, plane_uniform, resident_workgroups, resolve_workgroup_size, GpuLimits,
};
pub use crate::tensor::GpuTensor;
pub use crate::traits::CubeclFloat;
pub use cubecl_runtime::runtime::Runtime;
