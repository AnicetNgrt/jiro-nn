use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

use self::{sgd::SGD, momentum::Momentum};

pub mod sgd;
pub mod momentum;
// pub mod adam;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum Optimizers {
    SGD(SGD),
    Momentum(Momentum),
    // Adam(Adam),
}

impl Optimizers {
    pub fn update_learning_rate(&mut self, epoch: usize) -> f64 {
        match self {
            Optimizers::SGD(sgd) => sgd.update_learning_rate(epoch),
            Optimizers::Momentum(momentum) => momentum.update_learning_rate(epoch),
            // Optimizers::Adam(adam) => adam.update_learning_rate(epoch),
        }
    }

    pub fn update_weights(&mut self, epoch: usize, weights: &DMatrix<f64>, weights_gradient: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Optimizers::SGD(sgd) => sgd.update_weights(epoch, weights, weights_gradient),
            Optimizers::Momentum(momentum) => momentum.update_weights(epoch, weights, weights_gradient),
            // Optimizers::Adam(adam) => adam.update_weights(epoch, weights, weights_gradient),
        }
    }
}