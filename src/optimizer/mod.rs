use serde::{Serialize, Deserialize};

use crate::linalg::{Matrix};

use self::{sgd::SGD, momentum::Momentum, adam::Adam};

pub mod sgd;
pub mod momentum;
pub mod adam;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Optimizers {
    SGD(SGD),
    Momentum(Momentum),
    Adam(Adam),
}

impl Optimizers {
    pub fn update_parameters(&mut self, epoch: usize, parameters: &Matrix, parameters_gradient: &Matrix) -> Matrix {
        match self {
            Optimizers::SGD(sgd) => sgd.update_parameters(epoch, parameters, parameters_gradient),
            Optimizers::Momentum(momentum) => momentum.update_parameters(epoch, parameters, parameters_gradient),
            Optimizers::Adam(adam) => adam.update_parameters(epoch, parameters, parameters_gradient),
        }
    }

    pub fn adam_default() -> Optimizers {
        Optimizers::Adam(Adam::default())
    }
}