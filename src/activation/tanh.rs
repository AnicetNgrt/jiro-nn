use super::ActivationLayer;

pub fn new<const I: usize>() -> ActivationLayer<I> {
    ActivationLayer::new(nalgebra_glm::tanh, |m| {
        nalgebra_glm::tanh(m).map(|x| 1. - (x * x))
    })
}
