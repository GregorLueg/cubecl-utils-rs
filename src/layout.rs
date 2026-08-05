//! Row padding to a vectorisation boundary.

/// Lane count for vectorised loads.
///
/// Kernels index a row as `Vector<F, LINE_SIZE>`, so every row has to be a
/// whole number of lines. Four is what `vec4` gives on every backend worth
/// supporting.
pub const LINE_SIZE: usize = 4;

/// Round a dimensionality up to a whole number of lines.
///
/// ### Params
///
/// * `dim` - Original dimensionality
///
/// ### Returns
///
/// The smallest multiple of [`LINE_SIZE`] that is at least `dim`.
#[inline]
pub fn padded_dim(dim: usize) -> usize {
    dim.next_multiple_of(LINE_SIZE)
}

/// Pad rows to `dim_padded` by appending zeros to each.
///
/// ### Params
///
/// * `flat` - Flattened row-major data of size `n * dim`
/// * `n` - Number of rows
/// * `dim` - Original dimensionality
/// * `dim_padded` - Target dimensionality, must be at least `dim`
///
/// ### Returns
///
/// Padded flat data of size `n * dim_padded`.
///
/// ### Note
///
/// Padding with zeros is only sound for the metrics where a zero component
/// contributes nothing: dot products, squared Euclidean and the L2 norms
/// underneath cosine. It is not sound for anything that counts components.
pub fn pad_vectors<T: num_traits::Float>(
    flat: &[T],
    n: usize,
    dim: usize,
    dim_padded: usize,
) -> Vec<T> {
    let mut padded = vec![T::zero(); n * dim_padded];
    for i in 0..n {
        let src = &flat[i * dim..(i + 1) * dim];
        let dst = &mut padded[i * dim_padded..i * dim_padded + dim];
        dst.copy_from_slice(src);
    }
    padded
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_dim() {
        assert_eq!(padded_dim(0), 0);
        assert_eq!(padded_dim(1), 4);
        assert_eq!(padded_dim(4), 4);
        assert_eq!(padded_dim(5), 8);
        assert_eq!(padded_dim(128), 128);
    }

    #[test]
    fn test_pad_vectors_appends_zeros() {
        let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let padded = pad_vectors(&flat, 2, 3, 4);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn test_pad_vectors_noop_when_already_aligned() {
        let flat = vec![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(pad_vectors(&flat, 1, 4, 4), flat);
    }
}
