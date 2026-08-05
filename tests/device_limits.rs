//! Device-limit probes that need a real GPU.
//!
//! The point of these is not to assert a particular device's numbers, which
//! would fail on the next machine. It is to get the numbers *printed* on every
//! backend CI runs on, so that the assumptions baked into consumers of this
//! crate can be checked against something other than one developer's laptop.
//!
//! Run with `cargo test --features gpu-tests -- --nocapture`.

#![cfg(feature = "gpu-tests")]

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use cubecl_utils_rs::prelude::*;

/// Print every limit the crate reads, for whichever backend is active.
#[test]
fn test_report_device_limits() {
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let limits = GpuLimits::from_client(&client);

    println!("--- device limits ---");
    println!("max_shared_bytes   : {}", limits.max_shared_bytes);
    println!("max_cube_count     : {:?}", limits.max_cube_count);
    println!("max_units_per_cube : {}", limits.max_units_per_cube);
    println!("max_cube_dim       : {:?}", limits.max_cube_dim);
    println!("max_binding_bytes  : {}", limits.max_binding_bytes);
    println!(
        "plane_size         : {}..={}",
        limits.plane_size_min, limits.plane_size_max
    );
    println!("--- derived ---");
    println!(
        "resolve_workgroup_size(256) : {}",
        resolve_workgroup_size(256, &limits)
    );
    println!(
        "plane_uniform(32)           : {}",
        plane_uniform(32, &limits)
    );
    println!(
        "plane_uniform(64)           : {}",
        plane_uniform(64, &limits)
    );
    println!(
        "plane_partitions(256)       : {:?}",
        plane_partitions(256, &limits)
    );
    println!(
        "resident @ 32 KiB staging   : {}",
        resident_workgroups(32_768, &limits)
    );

    // Nothing here should ever be zero. A backend reporting one is a backend
    // this crate cannot size a dispatch for.
    assert!(limits.max_shared_bytes > 0, "no shared memory reported");
    assert!(limits.max_units_per_cube > 0, "no units per cube reported");
    assert!(limits.max_binding_bytes > 0, "no binding limit reported");
    assert!(limits.plane_size_min > 0, "no plane size reported");
    assert!(limits.plane_size_max >= limits.plane_size_min);
    let (cx, cy, cz) = limits.max_cube_count;
    assert!(cx > 0 && cy > 0 && cz > 0, "no cube count reported");
    let (dx, dy, dz) = limits.max_cube_dim;
    assert!(dx > 0 && dy > 0 && dz > 0, "no cube dim reported");
}

/// The geometry helpers must agree with the device they just read.
#[test]
fn test_derived_geometry_is_legal_on_this_device() {
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let limits = GpuLimits::from_client(&client);

    for total in [1u32, 1_000, 65_535, 65_536, 1_000_000, 10_000_000] {
        let (gx, gy) = grid_2d(total, &limits).expect("grid does not fit this device");
        assert!(checked_cube_count("probe", gx, gy, 1, &limits).is_ok());
        assert!(gx as u64 * gy as u64 >= total as u64);
    }

    let wg = resolve_workgroup_size(256, &limits);
    assert!(wg > 0 && wg <= limits.max_units_per_cube);
    assert!(wg <= limits.max_cube_dim.0);
}
