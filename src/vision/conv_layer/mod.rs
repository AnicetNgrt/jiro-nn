use std::fmt::Debug;

use crate::{layer::LearnableLayer, linalg::Scalar};

use super::{image::Image, image_layer::ImageLayer};

pub mod defaults;
pub mod dense_conv_layer;
pub mod direct_conv_layer;
pub mod avg_pooling_layer;
pub mod full_conv_layer;

pub trait ConvLayer: ImageLayer + LearnableLayer + Send + Debug {
    fn scale_kernels(&mut self, scale: Scalar);
}
