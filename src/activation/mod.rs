use std::fmt;

use crate::linalg::MatrixTrait;
use serde::{Serialize, Deserialize};

use crate::{layer::Layer, linalg::Matrix};

pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod linear;

pub type ActivationFn = fn(&Matrix) -> Matrix;

pub struct ActivationLayer {
    // i inputs = i outputs (it's just a map)
    input: Option<Matrix>,
    activation: ActivationFn,
    derivative: ActivationFn,
}

impl ActivationLayer {
    pub fn new(activation: ActivationFn, derivative: ActivationFn) -> Self {
        Self {
            input: None,
            activation,
            derivative,
        }
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = Some(input.clone());
        (self.activation)(&input)
    }

    fn backward(
        &mut self,
        _epoch: usize,
        output_gradient: Matrix
    ) -> Matrix {
        // ∂E/∂X = ∂E/∂Y ⊙ f'(X)
        let input = self.input.clone().unwrap();
        let fprime_x = (self.derivative)(&input);
        //println!("{:?} {:?}", output_gradient.shape(), fprime_x.shape());
        output_gradient.component_mul(&fprime_x)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    Linear
}

impl Activation {
    pub fn to_layer(&self) -> ActivationLayer {
        match self {
            Self::Linear => linear::new(),
            Self::Tanh => tanh::new(),
            Self::Sigmoid => sigmoid::new(),
            Self::ReLU => relu::new(),
        }
    }
}

impl fmt::Debug for ActivationLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Activation Layer")
    }
}
