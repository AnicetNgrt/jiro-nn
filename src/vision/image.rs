use crate::linalg::{Matrix, Scalar};

/// An image (or batched images) composed of Scalar n rows on m columns and c channels (with s samples if batched).
pub trait ImageTrait {
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

    /// Forward pass of a CNN 
    fn convolve_cnn(&self, kernels: &Self) -> Self;

    /// Gradient of the forward pass of a CNN
    /// 
    /// Returns (Input gradient, Filter gradient, Biases gradient)
    /// 
    /// It may seem like giving too much responsibility to the backend to do the whole gradient computation
    /// but it may be the most efficient way to do it. The backend can use its own specialized function
    /// such as the batched parallel and optimized CUDA implementation if CUDA is used.
    fn convolve_grad_cnn(&self, input: &Self, filter: &Self, output: &Self) -> (Self, Self, Self) where Self: Sized;

    fn flatten(&self) -> Matrix;

    /// Returns (nrow, ncol, nchan)
    fn image_dims(&self) -> (usize, usize, usize);

    /// Returns the amount of samples in the batch
    fn samples(&self) -> usize;

    /// Returns a full image (pixels + channels) for the given sample
    fn get_sample(&self, sample: usize) -> Self;

    /// Returns a single channel. Assumes the image contains only 1 sample.
    fn get_channel(&self, channel: usize) -> Self;

    fn join_channels(channels: Vec<Self>) -> Self where Self: Sized;

    fn join_samples(samples: Vec<Self>) -> Self where Self: Sized;
}
