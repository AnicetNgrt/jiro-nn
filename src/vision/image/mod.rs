use crate::linalg::{Matrix, Scalar};

#[cfg(feature = "arrayfire")]
pub mod arrayfire_image;
#[cfg(feature = "arrayfire")]
pub type Image = arrayfire_image::Image;

#[cfg(all(feature = "nalgebra", not(feature = "arrayfire")))]
pub mod nalgebra_image;
#[cfg(all(feature = "nalgebra", not(feature = "arrayfire")))]
pub type Image = nalgebra_image::Image;

#[cfg(all(
    feature = "ndarray",
    not(feature = "arrayfire"),
    not(feature = "nalgebra")
))]
pub mod ndarray_image;
#[cfg(all(
    feature = "ndarray",
    not(feature = "arrayfire"),
    not(feature = "nalgebra")
))]
pub type Image = ndarray_image::Image;

/// An image (or batched images) composed of Scalar n rows on m columns and c channels (with s samples if batched).
pub trait ImageTrait {
    fn zeros(nrow: usize, ncol: usize, nchan: usize, samples: usize) -> Self;

    fn constant(nrow: usize, ncol: usize, nchan: usize, samples: usize, value: Scalar) -> Self;

    fn random_uniform(
        nrow: usize,
        ncol: usize,
        nchan: usize,
        samples: usize,
        min: Scalar,
        max: Scalar,
    ) -> Self;

    fn random_normal(
        nrow: usize,
        ncol: usize,
        nchan: usize,
        samples: usize,
        mean: Scalar,
        stddev: Scalar,
    ) -> Self;

    fn from_fn<F>(nrows: usize, ncols: usize, nchan: usize, samples: usize, f: F) -> Self
    where
        F: FnMut(usize, usize, usize, usize) -> Scalar;

    /// `samples` has shape `(i, n)` where `n` is the number of samples, `i` is the number of pixels.
    ///
    /// Pixels are assumed to be in column-leading order with channels put in their entirety one after the other.
    fn from_samples(samples: &Matrix, channels: usize) -> Self;

    /// Adds the components of self and other. Assumes both images have the same pixel sizes and channels count.
    ///
    /// If other has less samples than self, it will add the first sample of other to all samples of self.
    fn component_add(&self, other: &Self) -> Self;

    /// Substracts the components of self and other. Assumes both images have the same pixel sizes and channels count.
    ///
    /// If other has less samples than self, it will substract the first sample of other to all samples of self.
    fn component_sub(&self, other: &Self) -> Self;

    /// Multiplies the components of self and other. Assumes both images have the same pixel sizes and channels count.
    ///
    /// If other has less samples than self, it will multiply the first sample of other to all samples of self.
    fn component_mul(&self, other: &Self) -> Self;

    /// Divides the components of self and other. Assumes both images have the same pixel sizes and channels count.
    ///
    /// If other has less samples than self, it will divide the first sample of other to all samples of self.
    fn component_div(&self, other: &Self) -> Self;

    fn scalar_add(&self, scalar: Scalar) -> Self;

    fn scalar_sub(&self, scalar: Scalar) -> Self;

    fn scalar_mul(&self, scalar: Scalar) -> Self;

    fn scalar_div(&self, scalar: Scalar) -> Self;

    fn cross_correlate(&self, kernels: &Self) -> Self;

    fn convolve_full(&self, kernels: &Self) -> Self;

    fn flatten(&self) -> Matrix;

    /// Returns (nrow, ncol, nchan)
    fn image_dims(&self) -> (usize, usize, usize);

    fn channels(&self) -> usize;

    /// Returns the amount of samples in the batch
    fn samples(&self) -> usize;

    /// Returns a full image (pixels + channels) for the given sample
    fn get_sample(&self, sample: usize) -> Self;

    /// Returns a single channel. Assumes the image contains only 1 sample.
    fn get_channel(&self, channel: usize) -> Self;

    /// Returns all the samples with only one channel.
    fn get_channel_across_samples(&self, channel: usize) -> Self;

    fn sum_samples(&self) -> Self;

    fn join_channels(channels: Vec<Self>) -> Self
    where
        Self: Sized;

    fn join_samples(samples: Vec<Self>) -> Self
    where
        Self: Sized;

    fn wrap(
        &self,
        ox: usize,
        oy: usize,
        wx: usize,
        wy: usize,
        sx: usize,
        sy: usize,
        px: usize,
        py: usize,
    ) -> Self;

    fn unwrap(&self, wx: usize, wy: usize, sx: usize, sy: usize, px: usize, py: usize) -> Self;

    fn tile(
        &self,
        repetitions_row: usize,
        repetitions_col: usize,
        repetitions_chan: usize,
        repetition_sample: usize,
    ) -> Self;

    fn square(&self) -> Self;

    fn sum(&self) -> Scalar;

    fn mean(&self) -> Scalar;

    fn mean_along(&self, dim: usize) -> Self;

    fn exp(&self) -> Self;

    fn maxof(&self, other: &Self) -> Self;

    fn sign(&self) -> Self;

    fn minof(&self, other: &Self) -> Self;

    fn sqrt(&self) -> Self;
}
