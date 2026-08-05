//! Shared trait bounds for element-generic kernels.

use cubecl::frontend::{CubePrimitive, Float};
use cubecl::CubeElement;

/// Float bound for kernels and tensors generic over the element type.
///
/// Blanket-implemented, so it is a naming convenience rather than a
/// restriction: any type CubeCL already accepts as a float element satisfies
/// it.
///
/// ### Note
///
/// `f64` satisfies this bound but is not universally available. WGSL has no
/// 64-bit float at all, so an `f64` kernel compiles here and then fails on the
/// wgpu backend; only the CUDA and HIP runtimes support it. Code that is
/// generic over the element type still needs to account for the element size
/// when it budgets shared memory or a binding.
pub trait CubeclFloat: Float + CubePrimitive + CubeElement {}

impl<T> CubeclFloat for T where T: Float + CubePrimitive + CubeElement {}
