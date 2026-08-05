//! Device limits and the dispatch geometry derived from them.
//!
//! Everything in here except [`GpuLimits::from_client`] is a pure function of
//! [`GpuLimits`]. That is deliberate: it means the behaviour on a device with
//! half the shared memory, a quarter of the units per cube or a smaller plane
//! can be asserted in a unit test on a machine that has none of those
//! properties.

use cubecl::prelude::*;

use crate::errors::CubeclUtilsErrors;

///////////////
// GpuLimits //
///////////////

/// Every device limit that dispatch geometry and staging decisions depend on.
///
/// Read once per client via [`GpuLimits::from_client`] and passed around as
/// data. Fields mirror `cubecl`'s `HardwareProperties` and
/// `MemoryDeviceProperties`.
///
/// ### Note
///
/// The values a backend reports are not uniform. Apple Silicon via wgpu gives
/// 32768 bytes of shared memory, a plane size pinned to 32/32 and a 4 GiB
/// binding limit. Integrated parts report as little as 16384 bytes of shared
/// memory, AMD reports a plane size of 64, and Intel reports a *range* because
/// the real value depends on register pressure and cannot be queried ahead of
/// time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLimits {
    /// Shared memory available to one workgroup, in bytes
    pub max_shared_bytes: usize,
    /// Maximum number of cubes per grid dimension, as `(x, y, z)`
    pub max_cube_count: (u32, u32, u32),
    /// Maximum number of units in a single cube
    pub max_units_per_cube: u32,
    /// Maximum extent of a cube per dimension, as `(x, y, z)`
    pub max_cube_dim: (u32, u32, u32),
    /// Largest single allocation or binding the device accepts, in bytes
    pub max_binding_bytes: u64,
    /// Smallest plane size the device may use
    pub plane_size_min: u32,
    /// Largest plane size the device may use
    pub plane_size_max: u32,
}

impl GpuLimits {
    /// Read the limits from a live compute client.
    ///
    /// `ComputeClient::properties()` is a field borrow rather than a device
    /// query, so this is cheap enough to call per allocation. It is still
    /// worth hoisting where several decisions share the same client.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client for the target device
    ///
    /// ### Returns
    ///
    /// A [`GpuLimits`] describing that device.
    ///
    /// ### Note
    ///
    /// The cube-count limit comes from the client properties rather than from
    /// `Runtime::max_cube_count()`. The latter is a per-backend constant: the
    /// wgpu implementation returns `u16::MAX` on every device regardless of
    /// what the adapter actually supports, which is a safe floor but discards
    /// headroom on hardware that allows more.
    pub fn from_client<R: Runtime>(client: &ComputeClient<R>) -> Self {
        let props = client.properties();
        let hw = &props.hardware;
        Self {
            max_shared_bytes: hw.max_shared_memory_size,
            max_cube_count: hw.max_cube_count,
            max_units_per_cube: hw.max_units_per_cube,
            max_cube_dim: hw.max_cube_dim,
            max_binding_bytes: props.memory.max_page_size,
            plane_size_min: hw.plane_size_min,
            plane_size_max: hw.plane_size_max,
        }
    }
}

///////////////////////
// Dispatch geometry //
///////////////////////

/// Split a flat cube count into a 2D grid bounded by `max_dim` per dimension.
///
/// The packing is x-fast row-major, so a kernel recovers its flat index with
/// `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X`. **That layout is a contract**, not
/// an implementation detail: kernel bodies across several crates decode it by
/// hand, and changing the shape silently corrupts every one of them.
///
/// ### Params
///
/// * `total_cubes` - Flat number of cubes the dispatch needs
/// * `max_dim` - Per-dimension limit to respect
///
/// ### Returns
///
/// `(x, y)` with `x * y >= total_cubes`, both within `max_dim`, or
/// `GridTooLarge` when no such pair exists.
///
/// ### Note
///
/// A `total_cubes` of zero is treated as one. A dispatch of nothing is a
/// caller-side no-op rather than an error, and the alternative was a division
/// by zero.
pub fn grid_2d_limited(total_cubes: u32, max_dim: u32) -> Result<(u32, u32), CubeclUtilsErrors> {
    let total = total_cubes.max(1);
    let limit = max_dim.max(1);

    let x = total.min(limit);
    let y = total.div_ceil(x);

    // y is unbounded by construction, and busts once total exceeds limit^2.
    if y > limit {
        return Err(CubeclUtilsErrors::GridTooLarge {
            total_cubes: total,
            max_dim: limit,
        });
    }

    Ok((x, y))
}

/// Split a flat cube count into a 2D grid within the device's x/y limits.
///
/// Convenience wrapper over [`grid_2d_limited`] using the smaller of the
/// device's x and y cube-count limits, so the result is valid on either axis.
///
/// ### Params
///
/// * `total_cubes` - Flat number of cubes the dispatch needs
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `(x, y)` with `x * y >= total_cubes`, or `GridTooLarge`.
pub fn grid_2d(total_cubes: u32, limits: &GpuLimits) -> Result<(u32, u32), CubeclUtilsErrors> {
    let (mx, my, _) = limits.max_cube_count;
    grid_2d_limited(total_cubes, mx.min(my))
}

/// Build a static cube count, checked against the device's per-dimension limit.
///
/// A dispatch that busts the limit is not a soft failure. The launch is
/// rejected on the CubeCL server thread, that thread dies, and every subsequent
/// call on the client returns an unrelated `CallError` from somewhere else
/// entirely. Catching it here turns that into a typed error naming the kernel.
///
/// ### Params
///
/// * `kernel` - Kernel name, for the error message only
/// * `x` - Requested cubes along x
/// * `y` - Requested cubes along y
/// * `z` - Requested cubes along z
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `CubeCount::Static(x, y, z)`, or `CubeCountExceeded` if any dimension is
/// over the device limit.
pub fn checked_cube_count(
    kernel: &'static str,
    x: u32,
    y: u32,
    z: u32,
    limits: &GpuLimits,
) -> Result<CubeCount, CubeclUtilsErrors> {
    let limit = limits.max_cube_count;
    if x > limit.0 || y > limit.1 || z > limit.2 {
        return Err(CubeclUtilsErrors::CubeCountExceeded {
            kernel,
            requested: (x, y, z),
            limit,
        });
    }
    Ok(CubeCount::Static(x, y, z))
}

//////////////////////
// Workgroup sizing //
//////////////////////

/// Make a preferred workgroup width legal on the target device.
///
/// Caps at `max_units_per_cube` and at the x extent of `max_cube_dim`, then
/// rounds down to a whole number of planes so no cube runs a partial SIMD
/// group. Rounding uses `plane_size_max`: on a device reporting a range, a
/// multiple of the largest candidate is a multiple of the smaller
/// power-of-two candidates too.
///
/// ### Params
///
/// * `preferred` - Workgroup width the caller would like
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// A legal workgroup width, never zero. It is a whole number of planes unless
/// a single plane is already wider than the device allows per cube, in which
/// case the cap wins and the caller gets a partial plane.
pub fn resolve_workgroup_size(preferred: u32, limits: &GpuLimits) -> u32 {
    let cap = limits.max_units_per_cube.min(limits.max_cube_dim.0).max(1);
    let wanted = preferred.clamp(1, cap);
    let plane = limits.plane_size_max.max(1);

    // A plane wider than the whole cube cannot be rounded to; the cap is the
    // harder constraint, so honour that and let the caller run partial.
    if plane > wanted {
        return wanted;
    }

    (wanted / plane) * plane
}

/////////////////////
// Plane viability //
/////////////////////

/// Whether a `wg_size`-wide workgroup is guaranteed to be exactly one plane.
///
/// This is the precondition for plane primitives that reduce across the whole
/// workgroup: `plane_max`, `plane_sum`, `plane_ballot` and friends operate on a
/// plane, so a workgroup straddling two of them silently reduces over half the
/// data. Both the reported min and max must equal the width, because a device
/// reporting a range gives no way to know which value it picked.
///
/// ### Params
///
/// * `wg_size` - Workgroup width the kernel will be launched at
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// True when the workgroup is exactly one plane on this device.
pub fn plane_uniform(wg_size: u32, limits: &GpuLimits) -> bool {
    limits.plane_size_min == wg_size && limits.plane_size_max == wg_size
}

/// How many whole planes a `wg_size`-wide workgroup divides into.
///
/// The weaker sibling of [`plane_uniform`], for kernels that reduce within each
/// plane and then combine the per-plane results through shared memory. The
/// plane size still has to be known exactly, but the workgroup may span several
/// of them.
///
/// Callers usually have their own bound on top of this, e.g. a shared-memory
/// scratch array sized for a maximum number of planes, or a minimum plane width
/// below which the plane path is not worth taking. Apply those to the returned
/// count.
///
/// ### Params
///
/// * `wg_size` - Workgroup width the kernel will be launched at
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `Some(n_planes)` when the device reports a single plane size that divides
/// `wg_size`, `None` otherwise.
pub fn plane_partitions(wg_size: u32, limits: &GpuLimits) -> Option<u32> {
    let plane = limits.plane_size_min;
    if plane == 0 || plane != limits.plane_size_max || wg_size == 0 {
        return None;
    }
    if !wg_size.is_multiple_of(plane) {
        return None;
    }
    Some(wg_size / plane)
}

/////////////////
// Allocations //
/////////////////

/// Check a single allocation against the device's per-binding size limit.
///
/// Two ceilings exist and they disagree: total device memory, and the largest
/// buffer that can be bound to one kernel argument. A wave of work can fit the
/// former while a single tensor busts the latter, and busting it is silent. On
/// wgpu the limit is `max_storage_buffer_binding_size`, which is 4 GiB on Apple
/// Silicon but as little as 128 MiB on parts that report only the WebGPU
/// defaults.
///
/// ### Params
///
/// * `requested` - Bytes the allocation needs
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `Ok(())` when it fits, `BindingTooLarge` otherwise.
pub fn fits_binding(requested: u64, limits: &GpuLimits) -> Result<(), CubeclUtilsErrors> {
    if requested > limits.max_binding_bytes {
        return Err(CubeclUtilsErrors::BindingTooLarge {
            requested,
            limit: limits.max_binding_bytes,
        });
    }
    Ok(())
}

///////////////////
// Shared memory //
///////////////////

/// Check a kernel's shared-memory footprint against the device budget.
///
/// Over-allocating shared memory is silent: the kernel does no work, writes
/// nothing, and reports no error. Anything whose `SharedMemory::new` argument
/// depends on a user-facing parameter (a neighbour count, an embedding
/// dimensionality, a graph degree) needs this before the launch.
///
/// ### Params
///
/// * `kernel` - Kernel name, for the error message only
/// * `requested` - Total bytes the kernel's shared allocations add up to
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `Ok(())` when it fits, `SharedMemoryExceeded` otherwise.
pub fn fits_shared_memory(
    kernel: &'static str,
    requested: usize,
    limits: &GpuLimits,
) -> Result<(), CubeclUtilsErrors> {
    if requested > limits.max_shared_bytes {
        return Err(CubeclUtilsErrors::SharedMemoryExceeded {
            kernel,
            requested,
            available: limits.max_shared_bytes,
        });
    }
    Ok(())
}

/// How many workgroups of a given shared-memory footprint stay resident.
///
/// Residency is the biggest lever on a latency-bound kernel, and it moves in
/// integer steps: a footprint of 22 KiB against a 32 KiB budget fits perfectly
/// and still runs at half the throughput of one that fits twice. Use this to
/// find which side of a threshold a candidate staging plan lands on.
///
/// ### Params
///
/// * `footprint_bytes` - Shared memory one workgroup allocates
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// Number of concurrently resident workgroups the shared-memory budget allows,
/// or 0 when a single workgroup does not fit.
pub fn resident_workgroups(footprint_bytes: usize, limits: &GpuLimits) -> usize {
    if footprint_bytes == 0 {
        return usize::MAX;
    }
    limits.max_shared_bytes / footprint_bytes
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Apple Silicon via wgpu, the machine everything was developed on.
    ///
    /// Verbatim from `tests/device_limits.rs` on an M-series part. The binding
    /// limit really is four bytes short of 4 GiB.
    fn apple() -> GpuLimits {
        GpuLimits {
            max_shared_bytes: 32_768,
            max_cube_count: (65_535, 65_535, 65_535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            max_binding_bytes: 4_294_967_292,
            plane_size_min: 32,
            plane_size_max: 32,
        }
    }

    /// A deliberately mean device: half the shared memory, a quarter of the
    /// units per cube, a 128 MiB binding limit and wave64.
    fn small() -> GpuLimits {
        GpuLimits {
            max_shared_bytes: 16_384,
            max_cube_count: (32_768, 1_024, 64),
            max_units_per_cube: 256,
            max_cube_dim: (256, 256, 64),
            max_binding_bytes: 128 * 1024 * 1024,
            plane_size_min: 64,
            plane_size_max: 64,
        }
    }

    // -- grid_2d --

    /// The packing `grid_2d` had before it learned about device limits. Kernel
    /// bodies in several crates decode this exact layout by hand.
    fn legacy_grid_2d(total_cubes: u32) -> (u32, u32) {
        let x = total_cubes.min(65535);
        let y = total_cubes.div_ceil(x);
        (x, y)
    }

    #[test]
    fn test_grid_2d_packing_matches_legacy() {
        for total in [
            1u32,
            2,
            31,
            32,
            1000,
            65_534,
            65_535,
            65_536,
            65_537,
            131_070,
            131_071,
            1_000_000,
            10_000_000,
            100_000_000,
        ] {
            assert_eq!(
                grid_2d_limited(total, 65_535).unwrap(),
                legacy_grid_2d(total),
                "packing drifted at total = {total}"
            );
        }
    }

    #[test]
    fn test_grid_2d_zero_does_not_panic() {
        // The old implementation divided by zero here.
        assert_eq!(grid_2d_limited(0, 65_535).unwrap(), (1, 1));
    }

    #[test]
    fn test_grid_2d_covers_and_fits() {
        for max_dim in [65_535u32, 32_768, 1024] {
            for total in [0u32, 1, 65_535, 65_536, 131_070, 10_000_000] {
                let capacity = max_dim as u64 * max_dim as u64;
                match grid_2d_limited(total, max_dim) {
                    Ok((x, y)) => {
                        assert!(
                            x <= max_dim && y <= max_dim,
                            "over limit at {total}/{max_dim}"
                        );
                        assert!(
                            x as u64 * y as u64 >= total.max(1) as u64,
                            "uncovered at {total}/{max_dim}"
                        );
                    }
                    Err(_) => assert!(
                        total as u64 > capacity,
                        "refused a grid that fits at {total}/{max_dim}"
                    ),
                }
            }
        }
    }

    #[test]
    fn test_grid_2d_errors_past_max_dim_squared() {
        // 1024^2 = 1_048_576, so one more cube than that cannot be packed.
        assert!(grid_2d_limited(1024 * 1024, 1024).is_ok());
        assert!(matches!(
            grid_2d_limited(1024 * 1024 + 1, 1024),
            Err(CubeclUtilsErrors::GridTooLarge { .. })
        ));
    }

    #[test]
    fn test_grid_2d_uses_smaller_of_x_and_y() {
        // small() reports 32768 on x but only 1024 on y, and the decomposition
        // has to be valid on whichever axis the caller assigns it to.
        let (x, y) = grid_2d(4096, &small()).unwrap();
        assert!(x <= 1024 && y <= 1024, "got {x}x{y}");
        assert!(x as u64 * y as u64 >= 4096);
    }

    // -- checked_cube_count --

    #[test]
    fn test_checked_cube_count_accepts_at_the_limit() {
        assert!(checked_cube_count("k", 65_535, 65_535, 65_535, &apple()).is_ok());
    }

    #[test]
    fn test_checked_cube_count_errors_per_axis() {
        let l = small();
        for (x, y, z) in [(32_769, 1, 1), (1, 32_769, 1), (1, 1, 65)] {
            assert!(
                matches!(
                    checked_cube_count("k", x, y, z, &l),
                    Err(CubeclUtilsErrors::CubeCountExceeded { .. })
                ),
                "accepted {x},{y},{z}"
            );
        }
    }

    // -- resolve_workgroup_size --

    #[test]
    fn test_resolve_workgroup_size_apple() {
        // 256 is already a whole number of 32-wide planes and fits the cap.
        assert_eq!(resolve_workgroup_size(256, &apple()), 256);
    }

    #[test]
    fn test_resolve_workgroup_size_caps_and_rounds() {
        // small(): cap 256, plane 64. 512 caps to 256, which is 4 planes.
        assert_eq!(resolve_workgroup_size(512, &small()), 256);
        // 200 caps to 200, rounds down to 3 planes = 192.
        assert_eq!(resolve_workgroup_size(200, &small()), 192);
    }

    #[test]
    fn test_resolve_workgroup_size_is_whole_planes_and_nonzero() {
        for plane in [8u32, 16, 32, 64] {
            for cap in [256u32, 512, 1024] {
                let l = GpuLimits {
                    max_units_per_cube: cap,
                    max_cube_dim: (cap, cap, cap),
                    plane_size_min: plane,
                    plane_size_max: plane,
                    ..apple()
                };
                for preferred in [1u32, 32, 100, 256, 4096] {
                    let wg = resolve_workgroup_size(preferred, &l);
                    assert!(wg > 0, "zero width at plane {plane}, cap {cap}");
                    assert!(wg <= cap, "over cap at plane {plane}, cap {cap}");
                    if preferred >= plane {
                        assert_eq!(wg % plane, 0, "partial plane at {plane}/{cap}/{preferred}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_resolve_workgroup_size_plane_wider_than_cap() {
        // wgpu invents 8/128 when a backend reports no subgroup info at all.
        let l = GpuLimits {
            max_units_per_cube: 64,
            max_cube_dim: (64, 64, 64),
            plane_size_min: 8,
            plane_size_max: 128,
            ..apple()
        };
        // Cannot round to a 128-wide plane inside a 64-unit cube; cap wins.
        assert_eq!(resolve_workgroup_size(256, &l), 64);
    }

    // -- plane viability --

    #[test]
    fn test_plane_uniform() {
        assert!(plane_uniform(32, &apple()));
        assert!(!plane_uniform(64, &apple()));
        assert!(plane_uniform(64, &small()));
    }

    #[test]
    fn test_plane_uniform_false_on_a_range() {
        let l = GpuLimits {
            plane_size_min: 8,
            plane_size_max: 32,
            ..apple()
        };
        assert!(
            !plane_uniform(32, &l),
            "a reported range is not a guarantee"
        );
    }

    #[test]
    fn test_plane_partitions() {
        assert_eq!(plane_partitions(256, &apple()), Some(8));
        assert_eq!(plane_partitions(256, &small()), Some(4));
        // Not a multiple of the plane size.
        assert_eq!(plane_partitions(96, &small()), None);
        // Device reports a range.
        let ranged = GpuLimits {
            plane_size_min: 8,
            plane_size_max: 32,
            ..apple()
        };
        assert_eq!(plane_partitions(256, &ranged), None);
    }

    // -- bindings --

    #[test]
    fn test_fits_binding_boundary() {
        let l = small();
        let limit = 128 * 1024 * 1024;
        assert!(fits_binding(limit, &l).is_ok());
        assert!(matches!(
            fits_binding(limit + 1, &l),
            Err(CubeclUtilsErrors::BindingTooLarge { .. })
        ));
    }

    #[test]
    fn test_fits_binding_across_element_sizes() {
        // The exhaustive-search transient in ann-search-rs: 8192 queries by a
        // 16384-row database chunk. 512 MiB for f32, 1 GiB for f64.
        let elems: u64 = 8192 * 16_384;
        for (bytes_per_elem, name) in [(4u64, "f32/u32"), (8, "f64")] {
            let bytes = elems * bytes_per_elem;
            assert!(fits_binding(bytes, &apple()).is_ok(), "{name} on apple");
            assert!(fits_binding(bytes, &small()).is_err(), "{name} on small");
        }
    }

    // -- shared memory --

    #[test]
    fn test_fits_shared_memory_boundary() {
        let l = apple();
        assert!(fits_shared_memory("k", 32_768, &l).is_ok());
        assert!(matches!(
            fits_shared_memory("k", 32_769, &l),
            Err(CubeclUtilsErrors::SharedMemoryExceeded { .. })
        ));
    }

    #[test]
    fn test_fits_shared_memory_apple_budget_busts_a_small_device() {
        // A staging plan tuned to 32 KiB is exactly what silently no-ops on a
        // 16 KiB part.
        assert!(fits_shared_memory("k", 20_000, &apple()).is_ok());
        assert!(fits_shared_memory("k", 20_000, &small()).is_err());
    }

    #[test]
    fn test_resident_workgroups() {
        let l = apple();
        assert_eq!(resident_workgroups(22_024, &l), 1);
        assert_eq!(resident_workgroups(13_320, &l), 2);
        assert_eq!(resident_workgroups(8_968, &l), 3);
        assert_eq!(resident_workgroups(40_000, &l), 0);
    }
}
