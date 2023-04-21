use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

use crate::learning_rate::{LearningRateSchedule, default_learning_rate};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SGD {
    #[serde(default="default_learning_rate")]
    learning_rate: LearningRateSchedule,
}

impl SGD {
    pub fn with_const_lr(learning_rate: f64) -> Self {
        Self {
            learning_rate: LearningRateSchedule::Constant(learning_rate),
        }
    }

    pub fn new(learning_rate: LearningRateSchedule) -> Self {
        Self {
            learning_rate,
        }
    }

    pub fn update_parameters(&mut self, epoch: usize, parameters: &DMatrix<f64>, parameters_gradient: &DMatrix<f64>) -> DMatrix<f64> {
        let lr = self.learning_rate.get_learning_rate(epoch);
        parameters - (lr * parameters_gradient)
    }
}