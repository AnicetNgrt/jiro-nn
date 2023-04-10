use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

use crate::learning_rate::LearningRateSchedule;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SGD {
    learning_rate: LearningRateSchedule,
    #[serde(skip)]
    cached_learning_rate: Option<f64>,
}

impl SGD {
    pub fn with_const_lr(learning_rate: f64) -> Self {
        Self {
            learning_rate: LearningRateSchedule::Constant(learning_rate),
            cached_learning_rate: None,
        }
    }

    pub fn new(learning_rate: LearningRateSchedule) -> Self {
        Self {
            learning_rate,
            cached_learning_rate: None,
        }
    }

    pub fn update_learning_rate(&mut self, epoch: usize) -> f64 {
        let lr = self.learning_rate.get_learning_rate(epoch);
        self.cached_learning_rate = Some(lr);
        lr
    }

    pub fn update_weights(&mut self, _epoch: usize, weights: &DMatrix<f64>, weights_gradient: &DMatrix<f64>) -> DMatrix<f64> {
        let learning_rate = self.cached_learning_rate.unwrap();
        weights - (learning_rate * weights_gradient)
    }
}