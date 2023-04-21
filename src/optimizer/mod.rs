use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

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
    pub fn update_parameters(&mut self, epoch: usize, parameters: &DMatrix<f64>, parameters_gradient: &DMatrix<f64>) -> DMatrix<f64> {
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