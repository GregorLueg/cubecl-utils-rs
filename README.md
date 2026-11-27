[![CI](https://github.com/GregorLueg/cubecl-utils-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/cubecl-utils-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/cubecl-utils-rs.svg)](https://crates.io/crates/cubecl-utils-rs)
[![docs.rs](https://img.shields.io/docsrs/cubecl-utils-rs)](https://docs.rs/cubecl-utils-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# cubecl-utils-rs

Shared [CubeCL](https://github.com/tracel-ai/cubecl) helpers: GPU tensors,
device-limit queries and validated dispatch geometry. On top of that, the
kernels that more than one downstream crate needs, so they live in one place
instead of being copied between `ann-search-rs`, `bixverse-rs` and
`manifolds-rs`. For now that's dense GEMM.

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
| `skinny_gemm` | `C = op(A) op(B)`, long reduction into a small output, split-K |
| `skinny_chunks`, `skinny_partial_elems` | split-K chunk count and the scratch size it needs |
| `tall_skinny_gemm`, `TSMM_K_MAX` | `C = A B` for a tall `A` and a small `B` |
| `CubeclUtilsErrors` | the one error type everything above returns |

The prelude also re-exports CubeCL's `Runtime` trait, so `R::client(&device)`
works without importing it from `cubecl-runtime`.

## GEMM kernels

Why not just use cubek? Its `Strategy::Auto` runs one thread per output
element with a serial reduction. Fine for square-ish products, bad for the two
shapes that keep turning up in single cell and embedding work:

- **Long reduction, small output.** A `[256, 256]` Gram matrix reduced over a
  million cells leaves nothing to parallelise over.
- **Tall operand against a small one.** `[n, k] * [k, s]` re-reads every row of
  the tall operand from global memory once per output column.

Some cubek strategies also ask for more shared memory than a 32 KiB device has
and get rejected outright.

`skinny_gemm` covers the first case. It is register-tiled and splits the
reduction (split-K) when there are too few output tiles to fill the device.
Either operand can be read transposed through a flag, so nothing is ever
transposed in memory, and `a` and `b` can be the same buffer:

```rust
// G = Y^T Y, with Y [n, s] row-major
let g = GpuTensor::<f32>::empty(vec![s, s], &client)?;
let partials =
    GpuTensor::<f32>::empty(vec![skinny_partial_elems(s, s, n).max(1)], &client)?;
skinny_gemm(&client, &y, &y, &g, &partials, s, s, n, true, false)?;
```

The partials buffer is caller-owned so a loop can reuse one scratch tensor
rather than allocating per call. When the reduction is not split it goes
untouched and the kernel writes straight into the output.

`tall_skinny_gemm` covers the second case. It stages rows of `A` in shared
memory once per workgroup, which caps the inner dimension at `TSMM_K_MAX`
(512). Anything wider returns `InnerDimTooLarge`; use `skinny_gemm` for those.

```rust
// C [n, s] = A [n, k] * B [k, s], k <= TSMM_K_MAX
tall_skinny_gemm(&client, &a, &b, &c, n, k, s)?;
```

Both are generic over `CubeclFloat` and follow the same rules as the rest of
the crate: every operand length is checked on the host (`BufferTooSmall`), and
shared memory and grid size are validated against `GpuLimits` before launch.

## CubeCL versions

CubeCL breaks its API between minor versions, and the types here go straight
into kernel launches, so pin the pair.

| cubecl-utils-rs | cubecl |
|---|---|
| 0.2, 0.3 | 0.11 |
| 0.0.1, 0.1 | 0.10 |

## Backends

The dependency on `cubecl` carries no backend feature, so the choice of wgpu,
CUDA, HIP or CPU is the consumer's. The test suite pulls in the CPU and wgpu
backends as dev-dependencies only.

Tests needing a real GPU sit behind the `gpu-tests` feature.

## Licence

MIT.
