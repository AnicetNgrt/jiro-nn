use crate::activation_layer::ActivationLayer;

pub struct TanhActivation<const I: usize>(ActivationLayer<I>);

impl<const I: usize> TanhActivation<I> {
    pub fn new() -> Self {
        Self(
            ActivationLayer::new(
                nalgebra_glm::tanh,
                |m| nalgebra_glm::tanh(m).map(|x| 1. - (x*x))
            )
        )
    }
}
