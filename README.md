[![CI](https://github.com/GregorLueg/cubecl-utils-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/cubecl-utils-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/cubecl-utils-rs.svg)](https://crates.io/crates/cubecl-utils-rs)
[![docs.rs](https://img.shields.io/docsrs/cubecl-utils-rs)](https://docs.rs/cubecl-utils-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# cubecl-utils-rs

Shared [CubeCL](https://github.com/tracel-ai/cubecl) helpers: GPU tensors,
device-limit queries and validated dispatch geometry. No algorithms, no
kernels, no backend preference.

## Why this exists

A CubeCL kernel dispatched with `launch_unchecked` that busts a device limit
does not fail loudly. Depending on which limit it busts it either does no work
and returns zeros, or it kills the CubeCL server thread so that an unrelated
later call reports a `CallError` with nothing pointing at the dispatch.

That makes device limits a correctness concern rather than a tuning one, and it
is easy to get wrong when every machine you develop on reports the same
numbers. Apple Silicon reports 32 KiB of shared memory, 65535 cubes per grid
dimension and a plane size of exactly 32. None of those are portable.

This crate keeps the answer to "what does this device allow" in one place.

## Design

Every limit decision is a **pure function of `GpuLimits`**:

```rust
use cubecl_utils_rs::prelude::*;

let limits = GpuLimits::from_client(&client);
let (gx, gy) = grid_2d(n_blocks, &limits)?;
let count = checked_cube_count("my_kernel", gx, gy, 1, &limits)?;
```

Only `GpuLimits::from_client` and the `GpuTensor` constructors touch a
client. Everything downstream takes limits as data, which is what makes
behaviour on a 16 KiB device testable on a machine that has 32 KiB.

Since CubeCL 0.11 the client is type-erased, so nothing here carries a runtime
generic. A tensor is `GpuTensor<F>`, not `GpuTensor<R, F>`:

```rust
let x = GpuTensor::<f32>::from_slice(&data, vec![n, dim], &client)?;
let out = GpuTensor::<f32>::empty(vec![n, k], &client)?;
// ... launch with x.into_tensor_arg(), out.into_tensor_arg() ...
let res = out.read(&client)?;
```

## What is in here

| item | purpose |
|---|---|
| `GpuTensor` | typed buffer handle; fallible constructors that check the per-binding limit |
| `GpuLimits` | shared memory, cube count, cube dim, units per cube, binding size, plane range |
| `grid_2d`, `grid_2d_limited` | flat cube count into a 2D grid within the device limit |
| `checked_cube_count` | validated `CubeCount`, naming the kernel on failure |
| `fits_binding` | allocation size against the per-binding limit |
| `fits_shared_memory` | a kernel's shared-memory footprint against the budget, naming the kernel on failure |
| `resident_workgroups` | how many workgroups of a given footprint stay resident |
| `resolve_workgroup_size` | preferred width capped and rounded to whole planes |
| `plane_uniform`, `plane_partitions` | guards before reaching for plane primitives |
| `pad_vectors`, `padded_dim`, `LINE_SIZE` | row padding to a vectorisation boundary |
| `CubeclFloat` | float bound for element-generic kernels |
| `CubeclUtilsErrors` | the one error type everything above returns |

The prelude also re-exports CubeCL's `Runtime` trait, so `R::client(&device)`
works without importing it from `cubecl-runtime`.

## CubeCL versions

CubeCL breaks its API between minor versions, and the types here go straight
into kernel launches, so pin the pair.

| cubecl-utils-rs | cubecl |
|---|---|
| 0.2 | 0.11 |
| 0.0.1, 0.1 | 0.10 |

## Backends

The dependency on `cubecl` carries no backend feature, so the choice of wgpu,
CUDA, HIP or CPU is the consumer's. The test suite pulls in the CPU and wgpu
backends as dev-dependencies only.

Tests needing a real GPU sit behind the `gpu-tests` feature.

## Licence

MIT.
