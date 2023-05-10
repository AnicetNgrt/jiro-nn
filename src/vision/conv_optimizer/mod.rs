use serde::{Deserialize, Serialize};

use self::{adam::ConvAdam, momentum::ConvMomentum, sgd::ConvSGD};

use super::image::Image;

pub mod adam;
pub mod momentum;
pub mod sgd;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConvOptimizers {
    ConvSGD(ConvSGD),
    ConvMomentum(ConvMomentum),
    ConvAdam(ConvAdam),
}

impl ConvOptimizers {
    pub fn update_parameters(
        &mut self,
        epoch: usize,
        parameters: &Image,
        parameters_gradient: &Image,
    ) -> Image {
        match self {
            ConvOptimizers::ConvSGD(sgd) => {
                sgd.update_parameters(epoch, parameters, parameters_gradient)
            }
            ConvOptimizers::ConvMomentum(momentum) => {
                momentum.update_parameters(epoch, parameters, parameters_gradient)
            }
            ConvOptimizers::ConvAdam(adam) => {
                adam.update_parameters(epoch, parameters, parameters_gradient)
            }
        }
    }
}

pub fn conv_adam() -> ConvOptimizers {
    ConvOptimizers::ConvAdam(ConvAdam::default())
}

pub fn conv_sgd() -> ConvOptimizers {
    ConvOptimizers::ConvSGD(ConvSGD::default())
}

pub fn conv_momentum() -> ConvOptimizers {
    ConvOptimizers::ConvMomentum(ConvMomentum::default())
}
