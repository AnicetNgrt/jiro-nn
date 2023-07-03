use super::ConvActivationLayer;
use crate::vision::{image::Image, image::ImageTrait};

fn tanh(m: &Image) -> Image {
    let exp = m.exp();
    let exp_neg = m.scalar_mul(-1.).exp();
    (exp.component_sub(&exp_neg)).component_div(&(exp.component_add(&exp_neg)))
}

fn tanh_prime(m: &Image) -> Image {
    let hbt = tanh(m);
    let hbt2 = &hbt.square();
    let ones = Image::constant(
        hbt.image_dims().0,
        hbt.image_dims().1,
        hbt.channels(),
        hbt.samples(),
        1.,
    );
    ones.component_sub(&hbt2)
}

pub fn new() -> ConvActivationLayer {
    ConvActivationLayer::new(tanh, tanh_prime)
}
