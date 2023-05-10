use super::image::Image;

pub mod dense_conv_layer;
pub mod full_conv_layer;

pub trait ConvLayer {
    fn forward(&mut self, input: Image) -> Image;
    fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image;
}