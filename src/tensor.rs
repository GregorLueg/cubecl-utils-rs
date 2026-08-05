//! GPU-resident tensor.

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubecl::zspace::{Shape, Strides};
use std::marker::PhantomData;

use crate::errors::CubeclUtilsErrors;
use crate::limits::{fits_binding, GpuLimits};

///////////////
// GpuTensor //
///////////////

/// GPU-resident tensor for use with CubeCL kernels.
pub struct GpuTensor<R: Runtime, F: CubeElement + Numeric> {
    /// Handle to the GPU buffer containing tensor data
    data: Handle,
    /// Dimensions of the tensor (e.g. `[n_rows, n_cols]`)
    shape: Vec<usize>,
    /// Memory strides for each dimension in row-major order
    strides: Vec<usize>,
    /// Phantom marker for the runtime type
    _r: PhantomData<R>,
    /// Phantom marker for the element type
    _f: PhantomData<F>,
}

impl<R: Runtime, F: CubeElement + Numeric> Clone for GpuTensor<R, F> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }
}

impl<R: Runtime, F: Numeric + CubeElement> GpuTensor<R, F> {
    /// Byte size of a tensor with the given shape.
    ///
    /// ### Params
    ///
    /// * `shape` - Dimensions of the tensor
    ///
    /// ### Returns
    ///
    /// Element count multiplied by the element size.
    fn byte_size(shape: &[usize]) -> u64 {
        (shape.iter().product::<usize>() * core::mem::size_of::<F>()) as u64
    }

    /// Create a tensor from CPU data.
    ///
    /// ### Params
    ///
    /// * `data` - Slice of values to upload
    /// * `shape` - Dimensions of the tensor
    /// * `client` - GPU compute client for memory allocation
    ///
    /// ### Returns
    ///
    /// A new tensor with the data copied to GPU memory, or `BindingTooLarge`
    /// when the allocation exceeds what this device binds in one go.
    pub fn from_slice(
        data: &[F],
        shape: Vec<usize>,
        client: &ComputeClient<R>,
    ) -> Result<Self, CubeclUtilsErrors> {
        fits_binding(Self::byte_size(&shape), &GpuLimits::from_client(client))?;

        let handle = client.create_from_slice(F::as_bytes(data));
        let strides = row_major_contiguous_strides(&shape).to_vec();
        Ok(Self {
            data: handle,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        })
    }

    /// Create an uninitialised tensor.
    ///
    /// ### Params
    ///
    /// * `shape` - Dimensions of the tensor
    /// * `client` - GPU compute client for memory allocation
    ///
    /// ### Returns
    ///
    /// A new tensor with allocated but uninitialised GPU memory, or
    /// `BindingTooLarge` when the allocation exceeds what this device binds in
    /// one go.
    ///
    /// ### Note
    ///
    /// The allocation returns quickly but its pages are not backed until
    /// something writes them. On a large buffer the first kernel write pays
    /// that fault and can cost more than the kernel itself, so prefer reusing
    /// one scratch tensor over allocating per call. See [`Self::reshaped_view`].
    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R>) -> Result<Self, CubeclUtilsErrors> {
        let size = Self::byte_size(&shape);
        fits_binding(size, &GpuLimits::from_client(client))?;

        let handle = client.empty(size as usize);
        let strides = row_major_contiguous_strides(&shape).to_vec();
        Ok(Self {
            data: handle,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        })
    }

    /// Convert to a `TensorArg` for kernel launches.
    ///
    /// ### Returns
    ///
    /// A `TensorArg` suitable for passing to CubeCL kernels. Vectorisation
    /// width is not set per tensor; it is passed once at launch as the argument
    /// for the kernel's `N: Size` generic.
    pub fn into_tensor_arg(&self) -> TensorArg<R> {
        unsafe {
            TensorArg::from_raw_parts(
                self.data.clone(),
                Strides::new(&self.strides),
                Shape::from(self.shape.clone()),
            )
        }
    }

    /// Read tensor data back to CPU.
    ///
    /// Consumes the tensor and transfers data from GPU to CPU memory.
    ///
    /// ### Params
    ///
    /// * `client` - GPU compute client for memory transfer
    ///
    /// ### Returns
    ///
    /// Vector of exactly [`Self::len`] elements.
    ///
    /// ### Note
    ///
    /// The read is truncated to the shape. That only matters for a tensor
    /// produced by [`Self::reshaped_view`], where the underlying allocation is
    /// larger than the view: the runtime hands back the whole binding, and
    /// returning that would silently include another view's data.
    pub fn read(self, client: &ComputeClient<R>) -> Result<Vec<F>, CubeclUtilsErrors> {
        let len = self.len();
        let bytes = client.read_one(self.data)?;
        let mut values = F::from_bytes(&bytes).to_vec();
        values.truncate(len);
        Ok(values)
    }

    /// Dimensions of the tensor.
    ///
    /// ### Returns
    ///
    /// Slice of the per-dimension extents, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements the underlying allocation holds.
    ///
    /// ### Returns
    ///
    /// Product of the shape dimensions.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the tensor holds no elements.
    ///
    /// ### Returns
    ///
    /// True if the element count is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reinterpret an existing allocation under a smaller shape.
    ///
    /// Shares the underlying buffer rather than allocating, so callers can keep
    /// one scratch tensor alive across several differently shaped uses. The
    /// first kernel write to a fresh allocation faults its pages in, which for
    /// a large buffer costs more than the kernel itself, so reuse is worth the
    /// sharp edge.
    ///
    /// ### Params
    ///
    /// * `shape` - New shape; its element count must not exceed the current one
    ///
    /// ### Returns
    ///
    /// A tensor sharing this one's buffer, with row-major strides for `shape`.
    ///
    /// ### Note
    ///
    /// The returned tensor aliases `self`. Writing through both concurrently is
    /// a data race the type system does not prevent here.
    pub fn reshaped_view(&self, shape: Vec<usize>) -> Self {
        debug_assert!(
            shape.iter().product::<usize>() <= self.len(),
            "reshaped_view would exceed the allocation"
        );
        let strides = row_major_contiguous_strides(&shape).to_vec();
        Self {
            data: self.data.clone(),
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Size of the tensor on the GPU, in bytes.
    ///
    /// ### Returns
    ///
    /// Element count multiplied by the element size.
    pub fn vram_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * std::mem::size_of::<F>()
    }

    /// Return the handle of the tensor.
    ///
    /// Escape hatch for crates that need to hand the raw buffer to their own
    /// kernels or to a library matmul without going through
    /// [`GpuTensor::into_tensor_arg`].
    ///
    /// ### Returns
    ///
    /// A reference to the underlying `Handle`.
    pub fn handle(&self) -> &Handle {
        &self.data
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_tensor_from_slice_and_read() {
        let client = CpuRuntime::client(&CpuDevice);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = GpuTensor::<CpuRuntime, f32>::from_slice(&data, vec![2, 3], &client).unwrap();

        assert_eq!(tensor.read(&client).unwrap(), data);
    }

    #[test]
    fn test_tensor_empty() {
        let client = CpuRuntime::client(&CpuDevice);

        let tensor = GpuTensor::<CpuRuntime, f32>::empty(vec![3, 4], &client).unwrap();

        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.len(), 12);
        assert!(!tensor.is_empty());
        assert_eq!(tensor.vram_bytes(), 48);
    }

    #[test]
    fn test_tensor_reshaped_view_shares_the_buffer() {
        let client = CpuRuntime::client(&CpuDevice);

        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = GpuTensor::<CpuRuntime, f32>::from_slice(&data, vec![3, 4], &client).unwrap();
        let view = tensor.reshaped_view(vec![2, 3]);

        assert_eq!(view.shape(), &[2, 3]);
        // The runtime hands back the whole 12-element binding; `read` truncates
        // to the view's own shape.
        assert_eq!(view.read(&client).unwrap(), &data[..6]);
    }

    #[test]
    fn test_tensor_rejects_an_oversized_binding() {
        let client = CpuRuntime::client(&CpuDevice);
        let limit = GpuLimits::from_client(&client).max_binding_bytes;

        // One element past what a single binding takes.
        let too_many = (limit / 4) as usize + 1;
        assert!(matches!(
            GpuTensor::<CpuRuntime, f32>::empty(vec![too_many], &client),
            Err(CubeclUtilsErrors::BindingTooLarge { .. })
        ));
    }
}
