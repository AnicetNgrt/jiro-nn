use super::image::Image;

pub trait ImageLayer {
    fn forward(&mut self, input: Image) -> Image;
    fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image;
}
