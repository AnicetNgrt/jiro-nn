use crate::vision::{image::ImageTrait, Image};
use super::ConvActivationLayer;

pub fn new() -> ConvActivationLayer {
    ConvActivationLayer::new(
        |m| {
            m.maxof(&Image::constant(
                m.image_dims().0,
                m.image_dims().1,
                m.image_dims().2,
                m.samples(),
                0.,
            ))
        },
        |m| {
            m.sign().maxof(&Image::constant(
                m.image_dims().0,
                m.image_dims().1,
                m.image_dims().2,
                m.samples(),
                0.,
            ))
        },
    )
}
