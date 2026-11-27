//! GEMM correctness against a host reference, on a real GPU.
//!
//! Run with `cargo test --release --features gpu-tests --test gemm`.

#![cfg(feature = "gpu-tests")]

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use cubecl_utils_rs::prelude::*;

/// The default wgpu client.
fn client() -> Client {
    <WgpuRuntime>::client(&WgpuDevice::default())
}

/// Deterministic signed values, so a dropped term cannot cancel out of sight.
fn build(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|t| (((t * 31 + salt * 17) % 23) as f32) * 0.1 - 1.1)
        .collect()
}

/// `C[rows, k] = op(A) op(B)` on the host in `f64`, with the kernel's indexing.
fn host_gemm(
    a: &[f32],
    b: &[f32],
    rows: usize,
    k: usize,
    len: usize,
    a_t: bool,
    b_t: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for c in 0..k {
            let mut acc = 0f64;
            for t in 0..len {
                let av = if a_t { a[t * rows + r] } else { a[r * len + t] };
                let bv = if b_t { b[c * len + t] } else { b[t * k + c] };
                acc += av as f64 * bv as f64;
            }
            out[r * k + c] = acc as f32;
        }
    }
    out
}

/// Relative tolerance floored at one, plus the all-zeros guard: a rejected
/// dispatch leaves the output untouched and reports nothing.
fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    assert!(
        got.iter().any(|v| v.abs() > 1e-6),
        "output is all zeros, the GPU did no work"
    );
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "index {i}: got {g} want {w}"
        );
    }
}

/// Run [`skinny_gemm`] on fresh operands and return GPU and host results.
fn run_skinny(rows: usize, k: usize, len: usize, a_t: bool, b_t: bool) -> (Vec<f32>, Vec<f32>) {
    let client = client();
    let a_host = build(rows * len, 1);
    let b_host = build(len * k, 2);

    let a_shape = if a_t {
        vec![len, rows]
    } else {
        vec![rows, len]
    };
    let b_shape = if b_t { vec![k, len] } else { vec![len, k] };
    let a = GpuTensor::<f32>::from_slice(&a_host, a_shape, &client).unwrap();
    let b = GpuTensor::<f32>::from_slice(&b_host, b_shape, &client).unwrap();
    let c = GpuTensor::<f32>::empty(vec![rows, k], &client).unwrap();
    let partials =
        GpuTensor::<f32>::empty(vec![skinny_partial_elems(rows, k, len).max(1)], &client).unwrap();

    skinny_gemm(&client, &a, &b, &c, &partials, rows, k, len, a_t, b_t).unwrap();

    let got = c.read(&client).unwrap();
    (got, host_gemm(&a_host, &b_host, rows, k, len, a_t, b_t))
}

/// All four orientation combinations.
const ORIENTATIONS: [(bool, bool); 4] =
    [(false, false), (true, false), (false, true), (true, true)];

/////////////////
// skinny_gemm //
/////////////////

// Ragged in every axis: every edge tile has a tail and the last staged step is
// short. All orientations, because each takes a different staging decode.
#[test]
fn test_skinny_gemm_ragged_tails() {
    for (a_t, b_t) in ORIENTATIONS {
        let (got, want) = run_skinny(200, 22, 37, a_t, b_t);
        assert_close(&got, &want, 1e-4);
    }
}

// A single output tile with a long reduction. Assert the arm is taken before
// trusting the result.
#[test]
fn test_skinny_gemm_split_k_path() {
    let (rows, k, len) = (64, 8, 4_096);
    assert!(skinny_chunks(rows, k, len) > 1, "expected the split-K arm");
    for (a_t, b_t) in ORIENTATIONS {
        let (got, want) = run_skinny(rows, k, len, a_t, b_t);
        assert_close(&got, &want, 1e-3);
    }
}

// Enough output tiles that the heuristic stops splitting and the kernel writes
// straight into the output.
#[test]
fn test_skinny_gemm_single_chunk_path() {
    let (rows, k, len) = (4_096, 128, 64);
    assert_eq!(
        skinny_chunks(rows, k, len),
        1,
        "expected the single-chunk arm"
    );
    for (a_t, b_t) in ORIENTATIONS {
        let (got, want) = run_skinny(rows, k, len, a_t, b_t);
        assert_close(&got, &want, 1e-4);
    }
}

// The narrowest output axis, where fifteen of sixteen staged columns are
// padding.
#[test]
fn test_skinny_gemm_rank_one() {
    let (got, want) = run_skinny(300, 1, 71, true, false);
    assert_close(&got, &want, 1e-4);
}

// Weight gradient of a 256-wide dense layer, `h^T delta`, and the backward
// product `delta W^T`: a square output over a long reduction, and a tall
// output against a transposed square operand.
#[test]
fn test_skinny_gemm_dense_layer_shapes() {
    let (got, want) = run_skinny(256, 256, 20_000, true, false);
    assert_close(&got, &want, 1e-3);
    let (got, want) = run_skinny(3_000, 256, 256, false, true);
    assert_close(&got, &want, 1e-4);
}

// `Y^T Y` with both operands bound to one buffer.
#[test]
fn test_skinny_gemm_aliased_gram() {
    let client = client();
    let (n, s) = (5_000, 30);
    let y_host = build(n * s, 3);
    let y = GpuTensor::<f32>::from_slice(&y_host, vec![n, s], &client).unwrap();
    let g = GpuTensor::<f32>::empty(vec![s, s], &client).unwrap();
    let partials =
        GpuTensor::<f32>::empty(vec![skinny_partial_elems(s, s, n).max(1)], &client).unwrap();

    skinny_gemm(&client, &y, &y, &g, &partials, s, s, n, true, false).unwrap();

    let got = g.read(&client).unwrap();
    assert_close(
        &got,
        &host_gemm(&y_host, &y_host, s, s, n, true, false),
        1e-3,
    );
}

#[test]
fn test_skinny_gemm_rejects_short_buffers() {
    let client = client();
    let (rows, k, len) = (64, 8, 4_096);
    assert!(skinny_chunks(rows, k, len) > 1);

    let a = GpuTensor::<f32>::empty(vec![rows, len], &client).unwrap();
    let b = GpuTensor::<f32>::empty(vec![len, k], &client).unwrap();
    let c = GpuTensor::<f32>::empty(vec![rows, k], &client).unwrap();
    let short = GpuTensor::<f32>::empty(vec![1], &client).unwrap();

    let err = skinny_gemm(&client, &a, &b, &c, &short, rows, k, len, false, false);
    assert!(matches!(
        err,
        Err(CubeclUtilsErrors::BufferTooSmall {
            buffer: "partials",
            ..
        })
    ));
    let err = skinny_gemm(&client, &a, &short, &c, &short, rows, k, len, false, false);
    assert!(matches!(
        err,
        Err(CubeclUtilsErrors::BufferTooSmall { buffer: "b", .. })
    ));
}

//////////////////////
// tall_skinny_gemm //
//////////////////////

// Ragged rows, an inner dimension that is not a multiple of the unroll, and a
// column count that is not a multiple of the column tile.
#[test]
fn test_tall_skinny_gemm_ragged() {
    let client = client();
    for (n, k_dim, s) in [(1_003, 130, 77), (5_000, 64, 256), (17, 3, 5)] {
        let a_host = build(n * k_dim, 4);
        let b_host = build(k_dim * s, 5);
        let a = GpuTensor::<f32>::from_slice(&a_host, vec![n, k_dim], &client).unwrap();
        let b = GpuTensor::<f32>::from_slice(&b_host, vec![k_dim, s], &client).unwrap();
        let c = GpuTensor::<f32>::empty(vec![n, s], &client).unwrap();

        tall_skinny_gemm(&client, &a, &b, &c, n, k_dim, s).unwrap();

        let got = c.read(&client).unwrap();
        let want = host_gemm(&a_host, &b_host, n, s, k_dim, false, false);
        assert_close(&got, &want, 1e-4);
    }
}

#[test]
fn test_tall_skinny_gemm_rejects_wide_inner_dim() {
    let client = client();
    let k_dim = TSMM_K_MAX as usize + 1;
    let a = GpuTensor::<f32>::empty(vec![8, k_dim], &client).unwrap();
    let b = GpuTensor::<f32>::empty(vec![k_dim, 4], &client).unwrap();
    let c = GpuTensor::<f32>::empty(vec![8, 4], &client).unwrap();

    assert!(matches!(
        tall_skinny_gemm(&client, &a, &b, &c, 8, k_dim, 4),
        Err(CubeclUtilsErrors::InnerDimTooLarge { .. })
    ));
}
