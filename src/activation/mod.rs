use nalgebra::DMatrix;

use crate::layer::Layer;

pub mod hbt;
pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub type ActivationFn = fn(&DMatrix<f64>) -> DMatrix<f64>;

pub struct ActivationLayer {
    // i inputs = i outputs (it's just a map)
    input: Option<DMatrix<f64>>,
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
    fn forward(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.input = Some(input.clone());
        (self.activation)(&input)
    }

    fn backward(
        &mut self,
        output_gradient: DMatrix<f64>,
        _learning_rate: f64,
    ) -> DMatrix<f64> {
        // ∂E/∂X = ∂E/∂Y ⊙ f'(X)
        let input = self.input.clone().unwrap();
        let fprime_x = (self.derivative)(&input);
        //println!("{:?} {:?}", output_gradient.shape(), fprime_x.shape());
        output_gradient.component_mul(&fprime_x)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    HyperbolicTangent,
}

impl Activation {
    pub fn to_layer(&self) -> ActivationLayer {
        match self {
            Self::Tanh => tanh::new(),
            Self::Sigmoid => sigmoid::new(),
            Self::ReLU => relu::new(),
            Self::HyperbolicTangent => hbt::new(),
        }
    }
}
