//! Dense GEMM for the shapes cubek handles badly: a long reduction into a small
//! output, and a tall operand against a small one.
//!
//! cubek's `Strategy::Auto` picks one thread per output element with a serial
//! reduction for both. A `[256, 256]` output reduced over a million rows then
//! has nothing to parallelise over, and a `[n, k] * [k, s]` product re-reads
//! each row of the tall operand from global memory once per output column.
//! Some of its strategies also ask for more shared memory than a 32 KiB device
//! offers and are rejected outright.
//!
//! * [`skinny_gemm`]: register-tiled `C = op(A) op(B)` with a split-K
//!   reduction when there are too few output tiles to fill the device. Covers
//!   `A^T B` (Gram matrices, weight gradients) and `A B^T` through comptime
//!   orientation flags, so no operand is ever transposed in memory.
//! * [`tall_skinny_gemm`]: `C = A B` for a tall `A` and a `B` small enough to
//!   stay cache-resident, staging rows of `A` once per workgroup.

// The `#[cube]` macro generates undocumented launcher structs and functions.
#![allow(missing_docs)]

use cubecl::prelude::*;

use crate::errors::CubeclUtilsErrors;
use crate::limits::{checked_cube_count, fits_shared_memory, grid_2d, GpuLimits};
use crate::tensor::GpuTensor;
use crate::traits::CubeclFloat;

////////////
// Consts //
////////////

/// Output rows per workgroup in [`skinny_gemm_partial`](fn@skinny_gemm_partial).
const SG_BM: u32 = 128;

/// Output columns per workgroup in [`skinny_gemm_partial`](fn@skinny_gemm_partial).
///
/// Fixed rather than tiered to the output width so there is one shader. Narrow
/// outputs waste part of the tile, but the waste is in the narrow axis and never
/// touches the traffic of the tall operand.
const SG_BN: u32 = 16;

/// Reduction steps staged per shared-memory round.
///
/// The staging footprint is `SG_BK * (SG_BM + SG_BN)` elements, so a small
/// value keeps several workgroups resident on a 32 KiB device. Occupancy beats
/// the better coalescing of a wider stage.
const SG_BK: u32 = 8;

/// Register tile height per thread.
const SG_RT_M: u32 = 4;

/// Register tile width per thread.
const SG_RT_N: u32 = 4;

/// Thread columns per workgroup.
const SG_TN: u32 = SG_BN / SG_RT_N;

/// Thread rows per workgroup.
const SG_TM: u32 = SG_BM / SG_RT_M;

/// Workgroup width of [`skinny_gemm_partial`](fn@skinny_gemm_partial): 128 threads, four planes of 32.
pub const SG_THREADS: u32 = SG_TM * SG_TN;

/// Output tiles the split-K heuristic aims to dispatch before it stops
/// splitting.
const SG_TARGET_CUBES: usize = 2048;

/// Hard cap on reduction chunks.
const SG_MAX_CHUNKS: usize = 256;

/// Shortest reduction worth giving its own chunk.
const SG_MIN_LEN_PER_CHUNK: usize = 512;

/// Ceiling on the partials buffer, in elements, implied by the other bounds.
///
/// There is no separate memory bound in [`skinny_chunks`] because one could
/// never fire. The split only happens when `tiles < SG_TARGET_CUBES`, and
/// `rows * k <= tiles * SG_BM * SG_BN`, so
///
/// ```text
/// chunks * rows * k  <=  ceil(TARGET / tiles) * tiles * BM * BN
///                    <=  TARGET * BM * BN  +  rows * k
///                    <=  2 * TARGET * BM * BN
/// ```
///
/// Only the test that enforces it reads it, hence the gate.
#[cfg(test)]
const SG_MAX_PARTIAL_ELEMS: usize = 2 * SG_TARGET_CUBES * (SG_BM * SG_BN) as usize;

/// Workgroup width of [`split_k_reduce`](fn@split_k_reduce).
const SG_REDUCE_WG: u32 = 256;

/// Rows of `A` handled per workgroup in [`tall_skinny_mm`](fn@tall_skinny_mm).
const TSMM_ROWS: u32 = 8;

/// Output columns handled per workgroup in [`tall_skinny_mm`](fn@tall_skinny_mm).
const TSMM_COLS: u32 = 32;

/// Largest inner dimension the shared-memory staging of [`tall_skinny_mm`](fn@tall_skinny_mm)
/// holds.
///
/// The footprint is `TSMM_ROWS * TSMM_K_MAX` elements, 16 KiB at `f32`, which
/// leaves two workgroups resident inside a 32 KiB budget.
pub const TSMM_K_MAX: u32 = 512;

/// Inner-loop unroll for [`tall_skinny_mm`](fn@tall_skinny_mm), to keep several loads of `b` in
/// flight rather than one.
const TSMM_UNROLL: u32 = 4;

/////////////
// Helpers //
/////////////

/// Split-K chunk count for one [`skinny_gemm`] product.
///
/// `wanted` is how many chunks it takes to fill the device given how few
/// output tiles there are; `by_len` stops a short reduction being cut into
/// slivers. Each binds for some reachable shape.
///
/// ### Params
///
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
///
/// ### Returns
///
/// Number of reduction chunks, at least one and at most `SG_MAX_CHUNKS`.
pub fn skinny_chunks(rows: usize, k: usize, len: usize) -> usize {
    let tiles = rows.div_ceil(SG_BM as usize) * k.div_ceil(SG_BN as usize);
    if tiles >= SG_TARGET_CUBES {
        return 1;
    }
    let wanted = SG_TARGET_CUBES.div_ceil(tiles.max(1));
    let by_len = (len / SG_MIN_LEN_PER_CHUNK).max(1);
    wanted.min(by_len).clamp(1, SG_MAX_CHUNKS)
}

/// Elements a caller-owned partials buffer needs for one [`skinny_gemm`]
/// product.
///
/// ### Params
///
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
///
/// ### Returns
///
/// `chunks * rows * k`, or zero when the product runs in a single chunk and
/// writes straight into its output.
pub fn skinny_partial_elems(rows: usize, k: usize, len: usize) -> usize {
    let chunks = skinny_chunks(rows, k, len);
    if chunks > 1 {
        chunks * rows * k
    } else {
        0
    }
}

/// Check that a buffer holds at least the elements a kernel will touch.
///
/// Kernels launched unchecked read and write past a short buffer without any
/// error, so this runs on the host before every dispatch.
///
/// ### Params
///
/// * `kernel` - Kernel name, for the error message only
/// * `buffer` - Buffer name, for the error message only
/// * `provided` - Elements the buffer holds
/// * `required` - Elements the kernel touches
///
/// ### Returns
///
/// `Ok(())` when the buffer is large enough, `BufferTooSmall` otherwise.
fn check_len(
    kernel: &'static str,
    buffer: &'static str,
    provided: usize,
    required: usize,
) -> Result<(), CubeclUtilsErrors> {
    if provided < required {
        return Err(CubeclUtilsErrors::BufferTooSmall {
            kernel,
            buffer,
            required,
            provided,
        });
    }
    Ok(())
}

/////////////
// Kernels //
/////////////

/// Partial `C[rows, k] = A[rows, len] * B[len, k]` over one chunk of the
/// reduction.
///
/// Both operands are read out of row-major buffers in either orientation:
///
/// ```text
/// a_transposed = false   A[row, t] = a[row * len + t]
/// a_transposed = true    A[row, t] = a[t * rows + row]
/// b_transposed = false   B[t, col] = b[t * k + col]
/// b_transposed = true    B[t, col] = b[col * len + t]
/// ```
///
/// The staging decode follows each flag, so adjacent threads always walk the
/// axis that is contiguous in the stored buffer. Getting that backwards costs
/// more than the register tile saves.
///
/// ### Params
///
/// * `a` - The `A` operand's backing buffer
/// * `b` - The `B` operand's backing buffer
/// * `out` - Output `[chunks, rows, k]`, or `[rows, k]` when there is one chunk
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
/// * `len_per_chunk` - Reduction steps per chunk; the last chunk is short
/// * `n_col_blocks` - `ceil(k / SG_BN)`, used to decode the packed z axis
/// * `a_transposed` - Which way to index `a` (comptime)
/// * `b_transposed` - Which way to index `b` (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row block, flattened because
///   the count is proportional to a data dimension
/// * `CUBE_POS_Z % n_col_blocks` -> column block,
///   `CUBE_POS_Z / n_col_blocks` -> reduction chunk. Both are bounded small,
///   so their product stays far below the per-dimension dispatch limit
#[cube(launch_unchecked)]
pub fn skinny_gemm_partial<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    out: &mut Tensor<F>,
    rows: u32,
    k: u32,
    len: u32,
    len_per_chunk: u32,
    n_col_blocks: u32,
    #[comptime] a_transposed: bool,
    #[comptime] b_transposed: bool,
) {
    let row_block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let col_block = CUBE_POS_Z % n_col_blocks;
    let chunk = CUBE_POS_Z / n_col_blocks;

    let i0 = row_block * SG_BM;
    let j0 = col_block * SG_BN;

    let mut sa = Shared::<[F]>::new_slice((SG_BK * SG_BM) as usize);
    let mut sb = Shared::<[F]>::new_slice((SG_BK * SG_BN) as usize);

    if i0 >= rows {
        terminate!();
    }

    let t_start = chunk * len_per_chunk;
    let mut t_end = t_start + len_per_chunk;
    if t_end > len {
        t_end = len;
    }

    let tid = UNIT_POS_X;
    let tm = tid / SG_TN;
    let tn = tid % SG_TN;

    let zero = F::new(0.0_f32);

    let mut acc = Array::<F>::new((SG_RT_M * SG_RT_N) as usize);
    #[unroll]
    for p in 0..SG_RT_M {
        #[unroll]
        for q in 0..SG_RT_N {
            acc[(p * SG_RT_N + q) as usize] = zero;
        }
    }

    let mut t = t_start;
    while t < t_end {
        // Out-of-range entries are zero-filled so the inner loop needs no
        // guard.
        let mut li = tid;
        while li < SG_BK * SG_BM {
            let kk = if a_transposed { li / SG_BM } else { li % SG_BK };
            let mm = if a_transposed { li % SG_BM } else { li / SG_BK };
            let tt = t + kk;
            let rr = i0 + mm;
            let mut v = zero;
            if tt < t_end && rr < rows {
                if a_transposed {
                    v = a[(tt * rows + rr) as usize];
                } else {
                    v = a[(rr * len + tt) as usize];
                }
            }
            sa[(kk * SG_BM + mm) as usize] = v;
            li += SG_THREADS;
        }

        let mut lj = tid;
        while lj < SG_BK * SG_BN {
            let kk = if b_transposed { lj % SG_BK } else { lj / SG_BN };
            let nn = if b_transposed { lj / SG_BK } else { lj % SG_BN };
            let tt = t + kk;
            let cc = j0 + nn;
            let mut v = zero;
            if tt < t_end && cc < k {
                if b_transposed {
                    v = b[(cc * len + tt) as usize];
                } else {
                    v = b[(tt * k + cc) as usize];
                }
            }
            sb[(kk * SG_BN + nn) as usize] = v;
            lj += SG_THREADS;
        }
        sync_cube();

        #[unroll]
        for kk in 0..SG_BK {
            let arow = kk * SG_BM + tm * SG_RT_M;
            let brow = kk * SG_BN + tn * SG_RT_N;
            let mut av = Array::<F>::new(SG_RT_M as usize);
            let mut bv = Array::<F>::new(SG_RT_N as usize);
            #[unroll]
            for p in 0..SG_RT_M {
                av[p as usize] = sa[(arow + p) as usize];
            }
            #[unroll]
            for q in 0..SG_RT_N {
                bv[q as usize] = sb[(brow + q) as usize];
            }
            #[unroll]
            for p in 0..SG_RT_M {
                #[unroll]
                for q in 0..SG_RT_N {
                    acc[(p * SG_RT_N + q) as usize] += av[p as usize] * bv[q as usize];
                }
            }
        }
        sync_cube();

        t += SG_BK;
    }

    let base = chunk * rows * k;
    #[unroll]
    for p in 0..SG_RT_M {
        let rr = i0 + tm * SG_RT_M + p;
        if rr < rows {
            #[unroll]
            for q in 0..SG_RT_N {
                let cc = j0 + tn * SG_RT_N + q;
                if cc < k {
                    out[(base + rr * k + cc) as usize] = acc[(p * SG_RT_N + q) as usize];
                }
            }
        }
    }
}

/// Sum split-K partials `[n_chunks, total]` into `[total]`.
///
/// ### Params
///
/// * `partials` - Per-chunk contributions `[n_chunks, total]`
/// * `out` - Output `[total]`
/// * `total` - Elements per chunk
/// * `n_chunks` - Number of chunks
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> workgroup index, flattened
///   because `total / SG_REDUCE_WG` grows with the output
/// * `UNIT_POS_X` -> element within the workgroup block
#[cube(launch_unchecked)]
pub fn split_k_reduce<F: Float>(
    partials: &Tensor<F>,
    out: &mut Tensor<F>,
    total: u32,
    n_chunks: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * SG_REDUCE_WG + UNIT_POS_X;
    if idx >= total {
        terminate!();
    }

    let mut acc = F::new(0.0_f32);
    let mut c = 0u32;
    while c < n_chunks {
        acc += partials[(c * total + idx) as usize];
        c += 1u32;
    }
    out[idx as usize] = acc;
}

/// `C[n, s] = A[n, k_dim] * B[k_dim, s]` for a tall `A` and a small `B`.
///
/// Each workgroup owns `TSMM_ROWS` rows and `TSMM_COLS` output columns. The
/// rows of `A` are staged in shared memory once and reused across the column
/// tile; `B` stays cache-resident and is read from global, where consecutive
/// threads hit consecutive columns and coalesce.
///
/// ### Params
///
/// * `a` - Left operand `[n, k_dim]`, row-major
/// * `b` - Right operand `[k_dim, s]`, row-major
/// * `c` - Output `[n, s]`, row-major
/// * `n` - Rows of `A`
/// * `k_dim` - Inner dimension, at most `TSMM_K_MAX`
/// * `s` - Output column count
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row block, flattened because
///   `n / TSMM_ROWS` passes the per-dimension dispatch limit at large `n`
/// * `CUBE_POS_Z` -> output column block
/// * `UNIT_POS_X` -> flattened `(row within block, column within block)`
#[cube(launch_unchecked)]
pub fn tall_skinny_mm<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    c: &mut Tensor<F>,
    n: u32,
    k_dim: u32,
    s: u32,
) {
    let tid = UNIT_POS_X;
    let tr = tid / TSMM_COLS;
    let tc = tid % TSMM_COLS;
    let n_threads = TSMM_ROWS * TSMM_COLS;

    let blk = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let row0 = blk * TSMM_ROWS;
    let col0 = CUBE_POS_Z * TSMM_COLS;

    // Stride is the runtime `k_dim`, so the tail of the allocation goes unused
    // when `k_dim < TSMM_K_MAX`.
    let mut sa = Shared::<[F]>::new_slice((TSMM_ROWS * TSMM_K_MAX) as usize);

    let stage = TSMM_ROWS * k_dim;
    let mut li = tid;
    while li < stage {
        let rr = li / k_dim;
        let cc = li % k_dim;
        let row = row0 + rr;
        if row < n {
            sa[li as usize] = a[(row * k_dim + cc) as usize];
        } else {
            sa[li as usize] = F::new(0.0_f32);
        }
        li += n_threads;
    }
    sync_cube();

    let row = row0 + tr;
    let col = col0 + tc;
    if row < n && col < s {
        let abase = tr * k_dim;
        let mut acc = F::new(0.0_f32);

        let mut i = 0u32;
        while i + TSMM_UNROLL <= k_dim {
            #[unroll]
            for u in 0..TSMM_UNROLL {
                acc += sa[(abase + i + u) as usize] * b[((i + u) * s + col) as usize];
            }
            i += TSMM_UNROLL;
        }
        while i < k_dim {
            acc += sa[(abase + i) as usize] * b[(i * s + col) as usize];
            i += 1u32;
        }

        c[(row * s + col) as usize] = acc;
    }
}

/////////////////
// Dispatchers //
/////////////////

/// Compute `C[rows, k] = A[rows, len] * B[len, k]` with either operand
/// read transposed.
///
/// Selects split-K via [`skinny_chunks`]. When it returns one, the kernel
/// accumulates the whole reduction and writes straight into `c`, no reduce is
/// launched and `partials` goes untouched. `a` and `b` may be the same buffer,
/// which is how `Y^T Y` is computed.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `a` - Backing buffer for `A`: `[rows, len]` row-major, or `[len, rows]`
///   when `a_transposed`
/// * `b` - Backing buffer for `B`: `[len, k]` row-major, or `[k, len]` when
///   `b_transposed`
/// * `c` - Output `[rows, k]` row-major
/// * `partials` - Caller-owned scratch of at least [`skinny_partial_elems`]
///   elements; unused when the reduction is not split
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
/// * `a_transposed` - Which way to index `a`
/// * `b_transposed` - Which way to index `b`
///
/// ### Returns
///
/// `Ok(())`, with `c` holding the product.
///
/// ### Errors
///
/// * `BufferTooSmall` if an operand, the output or `partials` holds fewer
///   elements than the product touches.
/// * `SharedMemoryExceeded` if the staged block does not fit the device.
/// * `CubeCountExceeded` if either grid is over the device's cube-count limit.
#[allow(clippy::too_many_arguments)]
pub fn skinny_gemm<F: CubeclFloat>(
    client: &Client,
    a: &GpuTensor<F>,
    b: &GpuTensor<F>,
    c: &GpuTensor<F>,
    partials: &GpuTensor<F>,
    rows: usize,
    k: usize,
    len: usize,
    a_transposed: bool,
    b_transposed: bool,
) -> Result<(), CubeclUtilsErrors> {
    const KERNEL: &str = "skinny_gemm_partial";

    check_len(KERNEL, "a", a.len(), rows * len)?;
    check_len(KERNEL, "b", b.len(), len * k)?;
    check_len(KERNEL, "c", c.len(), rows * k)?;

    let limits = GpuLimits::from_client(client);
    fits_shared_memory(
        KERNEL,
        (SG_BK * (SG_BM + SG_BN)) as usize * size_of::<F>(),
        &limits,
    )?;

    let n_chunks = skinny_chunks(rows, k, len);
    let split = n_chunks > 1;
    if split {
        check_len(KERNEL, "partials", partials.len(), n_chunks * rows * k)?;
    }
    let target = if split { partials } else { c };

    let len_per_chunk = len.div_ceil(n_chunks) as u32;
    let n_col_blocks = (k as u32).div_ceil(SG_BN);
    let row_blocks = (rows as u32).div_ceil(SG_BM);

    let (gx, gy) = grid_2d(row_blocks, &limits)?;
    let count = checked_cube_count(KERNEL, gx, gy, n_col_blocks * n_chunks as u32, &limits)?;

    unsafe {
        skinny_gemm_partial::launch_unchecked::<F>(
            client,
            count,
            CubeDim::new_1d(SG_THREADS),
            a.into_tensor_arg(),
            b.into_tensor_arg(),
            target.into_tensor_arg(),
            rows as u32,
            k as u32,
            len as u32,
            len_per_chunk,
            n_col_blocks,
            a_transposed,
            b_transposed,
        );
    }

    if split {
        let total = (rows * k) as u32;
        let (rx, ry) = grid_2d(total.div_ceil(SG_REDUCE_WG), &limits)?;
        let reduce_count = checked_cube_count("split_k_reduce", rx, ry, 1, &limits)?;
        unsafe {
            split_k_reduce::launch_unchecked::<F>(
                client,
                reduce_count,
                CubeDim::new_1d(SG_REDUCE_WG),
                partials.into_tensor_arg(),
                c.into_tensor_arg(),
                total,
                n_chunks as u32,
            );
        }
    }

    Ok(())
}

/// Compute `C[n, s] = A[n, k_dim] * B[k_dim, s]` for a tall `A` and a small
/// `B`, all row-major.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `a` - Left operand `[n, k_dim]`
/// * `b` - Right operand `[k_dim, s]`
/// * `c` - Output `[n, s]`
/// * `n` - Rows of `A`
/// * `k_dim` - Inner dimension, at most [`TSMM_K_MAX`]
/// * `s` - Output column count
///
/// ### Returns
///
/// `Ok(())`, with `c` holding the product.
///
/// ### Errors
///
/// * `InnerDimTooLarge` if `k_dim` exceeds [`TSMM_K_MAX`]; use
///   [`skinny_gemm`] for those.
/// * `BufferTooSmall` if an operand or the output is short.
/// * `SharedMemoryExceeded` if the row staging does not fit the device.
/// * `CubeCountExceeded` if the grid is over the device's cube-count limit.
#[allow(clippy::too_many_arguments)]
pub fn tall_skinny_gemm<F: CubeclFloat>(
    client: &Client,
    a: &GpuTensor<F>,
    b: &GpuTensor<F>,
    c: &GpuTensor<F>,
    n: usize,
    k_dim: usize,
    s: usize,
) -> Result<(), CubeclUtilsErrors> {
    const KERNEL: &str = "tall_skinny_mm";

    if k_dim > TSMM_K_MAX as usize {
        return Err(CubeclUtilsErrors::InnerDimTooLarge {
            kernel: KERNEL,
            requested: k_dim,
            max: TSMM_K_MAX as usize,
        });
    }
    check_len(KERNEL, "a", a.len(), n * k_dim)?;
    check_len(KERNEL, "b", b.len(), k_dim * s)?;
    check_len(KERNEL, "c", c.len(), n * s)?;

    let limits = GpuLimits::from_client(client);
    fits_shared_memory(
        KERNEL,
        (TSMM_ROWS * TSMM_K_MAX) as usize * size_of::<F>(),
        &limits,
    )?;

    let (gx, gy) = grid_2d((n as u32).div_ceil(TSMM_ROWS), &limits)?;
    let count = checked_cube_count(KERNEL, gx, gy, (s as u32).div_ceil(TSMM_COLS), &limits)?;

    unsafe {
        tall_skinny_mm::launch_unchecked::<F>(
            client,
            count,
            CubeDim::new_1d(TSMM_ROWS * TSMM_COLS),
            a.into_tensor_arg(),
            b.into_tensor_arg(),
            c.into_tensor_arg(),
            n as u32,
            k_dim as u32,
            s as u32,
        );
    }

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::grid_2d_limited;

    // Two workgroups have to stay resident, not merely fit: a footprint that
    // fits while leaving only one resident roughly halves throughput.
    #[test]
    fn test_gemm_shared_memory_within_budget() {
        let floor = 32 * 1024;
        let skinny = (SG_BK * (SG_BM + SG_BN)) as usize * size_of::<f32>();
        let tall = (TSMM_ROWS * TSMM_K_MAX) as usize * size_of::<f32>();
        for footprint in [skinny, tall] {
            assert!(footprint <= floor, "over the shared-memory floor");
            assert!(
                floor / footprint >= 2,
                "only one workgroup would be resident"
            );
        }
        assert_eq!(SG_THREADS, 128);
        assert_eq!(SG_BM % SG_RT_M, 0);
        assert_eq!(SG_BN % SG_RT_N, 0);
    }

    // The row-block count is proportional to a data dimension and the z axis
    // carries column blocks times chunks. Both are checked past the point where
    // a flat grid would bust the per-dimension dispatch limit.
    #[test]
    fn test_skinny_gemm_grid_within_dispatch_limit() {
        let max = u32::from(u16::MAX);
        assert!((16_000_000u32).div_ceil(SG_BM) > max);

        for rows in [1_000_000usize, 16_000_000, 100_000_000] {
            let blocks = (rows as u32).div_ceil(SG_BM);
            let (gx, gy) = grid_2d_limited(blocks, max).unwrap();
            assert!(gx <= max && gy <= max);
            assert!(gx as u64 * gy as u64 >= blocks as u64, "grid misses work");
        }

        // A split only happens below SG_TARGET_CUBES tiles and asks for at most
        // ceil(TARGET / tiles) chunks, so the packed z axis stays below
        // 2 * TARGET whatever the shape. Unsplit, z is just the column blocks,
        // and checked_cube_count catches an absurdly wide output.
        for &(rows, k) in &[
            (1usize, 1usize),
            (128, 16),
            (256, 256),
            (1, 4096),
            (500, 10),
        ] {
            for &len in &[600usize, 1_000_000, 1_000_000_000] {
                let chunks = skinny_chunks(rows, k, len);
                if chunks > 1 {
                    let z = (k as u32).div_ceil(SG_BN) * chunks as u32;
                    assert!(z <= 2 * SG_TARGET_CUBES as u32, "{rows}x{k}: z = {z}");
                }
            }
        }
    }

    // Every bound in the split-K heuristic has to bind for some reachable
    // shape, or it is dead code pretending to be a guard.
    #[test]
    fn test_skinny_chunks_each_bound_binds() {
        assert_eq!(skinny_chunks(1_000_000, 128, 4_096), 1);
        assert!(skinny_chunks(128, 16, 1_000_000) > 1);
        assert_eq!(skinny_chunks(128, 16, 600), 1);
        assert_eq!(skinny_chunks(128, 16, 1_000_000_000), SG_MAX_CHUNKS);

        for &(rows, k) in &[
            (1usize, 1usize),
            (128, 16),
            (256, 256),
            (500, 10),
            (3_000, 30),
            (50_000, 30),
            (200_000, 128),
            (1_000_000, 64),
        ] {
            for &len in &[600usize, 4_096, 1_000_000, 1_000_000_000] {
                let elems = skinny_partial_elems(rows, k, len);
                assert!(
                    elems <= SG_MAX_PARTIAL_ELEMS,
                    "{rows}x{k} over {len}: partials {elems} above the implied ceiling"
                );
            }
        }
    }

    #[test]
    fn test_skinny_partial_elems_is_zero_when_unsplit() {
        assert_eq!(skinny_partial_elems(1_000_000, 128, 4_096), 0);
        let (rows, k, len) = (128usize, 16usize, 1_000_000usize);
        assert_eq!(
            skinny_partial_elems(rows, k, len),
            skinny_chunks(rows, k, len) * rows * k
        );
    }
}
