use super::ConvActivationLayer;
use crate::vision::{image::Image, image::ImageTrait};

pub fn new() -> ConvActivationLayer {
    ConvActivationLayer::new(
        |m| {
            m.maxof(&Image::constant(
                m.image_dims().0,
                m.image_dims().1,
                m.channels(),
                m.samples(),
                0.,
            ))
        },
        |m| {
            m.sign().maxof(&Image::constant(
                m.image_dims().0,
                m.image_dims().1,
                m.channels(),
                m.samples(),
                0.,
            ))
        },
    )
}
