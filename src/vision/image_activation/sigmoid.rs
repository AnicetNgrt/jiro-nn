use super::ConvActivationLayer;
use crate::vision::{image::Image, image::ImageTrait};

fn sigmoid(m: &Image) -> Image {
    let exp_neg = m.scalar_mul(-1.).exp();
    let ones = Image::constant(
        m.image_dims().0,
        m.image_dims().1,
        m.channels(),
        m.samples(),
        1.,
    );
    ones.component_div(&(ones.component_add(&exp_neg)))
}

fn sigmoid_prime(m: &Image) -> Image {
    let sig = sigmoid(m);
    let ones = Image::constant(
        m.image_dims().0,
        m.image_dims().1,
        m.channels(),
        m.samples(),
        1.,
    );
    sig.component_mul(&(ones.component_sub(&sig)))
}

pub fn new() -> ConvActivationLayer {
    ConvActivationLayer::new(sigmoid, sigmoid_prime)
}
