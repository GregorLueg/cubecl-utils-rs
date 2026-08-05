# News

## 0.1.0

Official release after internal testing.

## 0.0.1

Initial release. The shared CubeCL primitives previously living inside
`ann-search-rs`, plus the device-limit handling that was scattered across
`bixverse-rs` and `manifolds-rs`.

**Features**

- `GpuTensor`, a thin typed wrapper over a CubeCL buffer handle. `empty` and
  `from_slice` are fallible and check the requested allocation against the
  device's per-binding limit before asking for it.
- `GpuLimits`, one struct carrying every device limit the dispatch geometry
  needs, read from `ComputeClient::properties()` in a single place.
- `grid_2d` and `grid_2d_limited`, which decompose a flat cube count into a 2D
  grid within the device's per-dimension limit.
- `checked_cube_count`, which validates a dispatch against
  `Runtime::max_cube_count()` and names the kernel in the error.
- `resolve_workgroup_size`, which caps a preferred workgroup width at
  `max_units_per_cube` and rounds it to a whole number of planes.
- `plane_uniform` and `plane_partitions`, the guards a kernel needs before using
  plane primitives. A workgroup straddling two planes gives silently wrong
  answers, so both take the workgroup width and the device's reported plane
  range rather than assuming 32.
- `pad_vectors` and `LINE_SIZE` for row padding to a vectorisation boundary.
- `CubeclFloat`, the float bound for kernels generic over the element type.
- Every limit decision is a pure function of `GpuLimits`, so the behaviour on
  devices smaller than the development machine is testable without owning one.
