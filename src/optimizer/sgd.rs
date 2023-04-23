use serde::{Serialize, Deserialize};

use crate::{learning_rate::{LearningRateSchedule, default_learning_rate}, linalg::{Matrix, MatrixTrait}};

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

    pub fn update_parameters(&mut self, epoch: usize, parameters: &Matrix, parameters_gradient: &Matrix) -> Matrix {
        let lr = self.learning_rate.get_learning_rate(epoch);
        parameters.component_sub(&parameters_gradient.scalar_mul(lr))
    }
}