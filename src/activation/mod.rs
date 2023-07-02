use std::fmt;

use crate::linalg::MatrixTrait;
use serde::{Deserialize, Serialize};

use crate::{layer::Layer, linalg::Matrix};

pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

pub type ActivationFn = fn(&Matrix) -> Matrix;
pub type GradDepActivationFn = fn(&Matrix, &Matrix) -> Matrix;

pub enum ActivationFnPrime {
    ActivationFn(ActivationFn),
    GradDepActivationFn(GradDepActivationFn),
}

pub struct ActivationLayer {
    // i inputs = i outputs (it's just a map)
    input: Option<Matrix>,
    output: Option<Matrix>,
    activation: ActivationFn,
    derivative: ActivationFnPrime,
}

impl ActivationLayer {
    pub fn new(activation: ActivationFn, derivative: ActivationFn) -> Self {
        Self {
            input: None,
            output: None,
            activation,
            derivative: ActivationFnPrime::ActivationFn(derivative),
        }
    }

    pub fn new_grad_dep(activation: ActivationFn, derivative: GradDepActivationFn) -> Self {
        Self {
            input: None,
            output: None,
            activation,
            derivative: ActivationFnPrime::GradDepActivationFn(derivative),
        }
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = Some(input.clone());
        let output = (self.activation)(&input);
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, _epoch: usize, output_gradient: Matrix) -> Matrix {
        match self.derivative {
            ActivationFnPrime::ActivationFn(f) => {
                // ∂E/∂X = ∂E/∂Y ⊙ f'(X)
                let input = self.input.clone().unwrap();
                let fprime_x = (f)(&input);
                output_gradient.component_mul(&fprime_x)
            },
            ActivationFnPrime::GradDepActivationFn(f) => {
                let output = self.output.clone().unwrap();
                (f)(&output, &output_gradient)
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    Linear,
    Softmax
}

impl Activation {
    pub fn to_layer(&self) -> ActivationLayer {
        match self {
            Self::Linear => linear::new(),
            Self::Tanh => tanh::new(),
            Self::Sigmoid => sigmoid::new(),
            Self::ReLU => relu::new(),
            Self::Softmax => softmax::new(),
        }
    }
}

impl fmt::Debug for ActivationLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Activation Layer")
    }
}
