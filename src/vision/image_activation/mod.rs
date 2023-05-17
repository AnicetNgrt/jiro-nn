use std::fmt;

use serde::{Deserialize, Serialize};

use super::{image::Image, image::ImageTrait};

pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub type ConvActivationFn = fn(&Image) -> Image;

pub struct ConvActivationLayer {
    input: Option<Image>,
    activation: ConvActivationFn,
    derivative: ConvActivationFn,
}

impl ConvActivationLayer {
    pub fn new(activation: ConvActivationFn, derivative: ConvActivationFn) -> Self {
        Self {
            input: None,
            activation,
            derivative,
        }
    }

    pub fn forward(&mut self, input: Image) -> Image {
        self.input = Some(input.clone());
        (self.activation)(&input)
    }

    pub fn backward(&mut self, _epoch: usize, output_gradient: Image) -> Image {
        let input = self.input.clone().unwrap();
        let fprime_x = (self.derivative)(&input);
        output_gradient.component_mul(&fprime_x)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConvActivation {
    ConvTanh,
    ConvSigmoid,
    ConvReLU,
    ConvLinear,
}

impl ConvActivation {
    pub fn to_layer(&self) -> ConvActivationLayer {
        match self {
            Self::ConvLinear => linear::new(),
            Self::ConvTanh => tanh::new(),
            Self::ConvSigmoid => sigmoid::new(),
            Self::ConvReLU => relu::new(),
        }
    }
}

impl fmt::Debug for ConvActivationLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Convolutional Activation Layer")
    }
}
