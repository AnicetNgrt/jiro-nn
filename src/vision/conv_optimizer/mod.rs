use serde::{Serialize, Deserialize};

use self::{sgd::SGD, momentum::Momentum, adam::Adam};

use super::Image;

pub mod sgd;
pub mod momentum;
pub mod adam;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConvOptimizers {
    SGD(SGD),
    Momentum(Momentum),
    Adam(Adam),
}

impl ConvOptimizers {
    pub fn update_parameters(&mut self, epoch: usize, parameters: &Image, parameters_gradient: &Image) -> Image {
        match self {
            ConvOptimizers::SGD(sgd) => sgd.update_parameters(epoch, parameters, parameters_gradient),
            ConvOptimizers::Momentum(momentum) => momentum.update_parameters(epoch, parameters, parameters_gradient),
            ConvOptimizers::Adam(adam) => adam.update_parameters(epoch, parameters, parameters_gradient),
        }
    }
}

pub fn adam() -> ConvOptimizers {
    ConvOptimizers::Adam(Adam::default())
}

pub fn sgd() -> ConvOptimizers {
    ConvOptimizers::SGD(SGD::default())
}

pub fn momentum() -> ConvOptimizers {
    ConvOptimizers::Momentum(Momentum::default())
}