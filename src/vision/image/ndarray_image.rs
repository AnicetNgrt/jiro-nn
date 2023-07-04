use crate::linalg::{Matrix, Scalar};

use super::ImageTrait;

#[derive(Clone, Debug)]
pub struct Image(usize);

#[allow(unused_variables)]
impl ImageTrait for Image {
    fn zeros(nrow: usize, ncol: usize, nchan: usize, samples: usize) -> Self {
        unimplemented!()
    }

    fn constant(nrow: usize, ncol: usize, nchan: usize, samples: usize, value: Scalar) -> Self {
        unimplemented!()
    }

    fn random_uniform(
        nrow: usize,
        ncol: usize,
        nchan: usize,
        samples: usize,
        min: Scalar,
        max: Scalar,
    ) -> Self {
        unimplemented!()
    }

    fn random_normal(
        nrow: usize,
        ncol: usize,
        nchan: usize,
        samples: usize,
        mean: Scalar,
        stddev: Scalar,
    ) -> Self {
        unimplemented!()
    }

    fn from_fn<F>(nrows: usize, ncols: usize, nchan: usize, samples: usize, f: F) -> Self
    where
        F: FnMut(usize, usize, usize, usize) -> Scalar,
    {
        unimplemented!()
    }

    fn from_samples(samples: &Matrix, channels: usize) -> Self {
        unimplemented!()
    }

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
    ) -> Self {
        unimplemented!()
    }

    fn unwrap(&self, wx: usize, wy: usize, sx: usize, sy: usize, px: usize, py: usize) -> Self {
        unimplemented!()
    }

    fn tile(
        &self,
        repetitions_row: usize,
        repetitions_col: usize,
        repetitions_chan: usize,
        repetition_sample: usize,
    ) -> Self {
        unimplemented!()
    }

    fn component_add(&self, other: &Self) -> Self {
        unimplemented!()
    }

    fn component_sub(&self, other: &Self) -> Self {
        unimplemented!()
    }

    fn component_mul(&self, other: &Self) -> Self {
        unimplemented!()
    }

    fn component_div(&self, other: &Self) -> Self {
        unimplemented!()
    }

    fn scalar_add(&self, scalar: Scalar) -> Self {
        unimplemented!()
    }

    fn scalar_sub(&self, scalar: Scalar) -> Self {
        unimplemented!()
    }

    fn scalar_mul(&self, scalar: Scalar) -> Self {
        unimplemented!()
    }

    fn scalar_div(&self, scalar: Scalar) -> Self {
        unimplemented!()
    }

    fn cross_correlate(&self, kernels: &Self) -> Self {
        unimplemented!()
    }

    fn convolve_full(&self, kernels: &Self) -> Self {
        unimplemented!()
    }

    fn flatten(&self) -> Matrix {
        unimplemented!()
    }

    fn image_dims(&self) -> (usize, usize, usize) {
        unimplemented!()
    }

    fn channels(&self) -> usize {
        unimplemented!()
    }

    fn samples(&self) -> usize {
        unimplemented!()
    }

    fn get_sample(&self, sample: usize) -> Self {
        unimplemented!()
    }

    fn get_channel(&self, channel: usize) -> Self {
        unimplemented!()
    }

    fn get_channel_across_samples(&self, channel: usize) -> Self {
        unimplemented!()
    }

    fn sum_samples(&self) -> Self {
        unimplemented!()
    }

    fn join_channels(channels: Vec<Self>) -> Self {
        unimplemented!()
    }

    fn join_samples(samples: Vec<Self>) -> Self {
        unimplemented!()
    }

    fn square(&self) -> Self {
        unimplemented!()
    }

    fn sum(&self) -> Scalar {
        unimplemented!()
    }

    fn mean(&self) -> Scalar {
        unimplemented!()
    }

    fn mean_along(&self, dim: usize) -> Self {
        unimplemented!()
    }

    fn exp(&self) -> Self {
        unimplemented!()
    }

    fn maxof(&self, other: &Self) -> Self {
        unimplemented!()
    }

    fn sign(&self) -> Self {
        unimplemented!()
    }

    fn minof(&self, other: &Self) -> Self {
        unimplemented!()
    }

    fn sqrt(&self) -> Self {
        unimplemented!()
    }
}
