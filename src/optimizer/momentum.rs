use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::learning_rate::{LearningRateSchedule, default_learning_rate};

fn default_momentum() -> f64 {
    0.9
}

// https://arxiv.org/pdf/1207.0580.pdf
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Momentum {
    #[serde(default="default_momentum")]
    momentum: f64,
    #[serde(default="default_learning_rate")]
    learning_rate: LearningRateSchedule,
    #[serde(skip)]
    v: Option<DMatrix<f64>>,
}

impl Momentum {
    pub fn new(learning_rate: LearningRateSchedule, momentum: f64) -> Self {
        Self {
            v: None,
            momentum,
            learning_rate,
        }
    }

    pub fn update_parameters(&mut self, epoch: usize, parameters: &DMatrix<f64>, parameters_gradient: &DMatrix<f64>) -> DMatrix<f64> {
        let lr = self.learning_rate.get_learning_rate(epoch);

        let v = if let Some(v) = self.v.clone() {
            v
        } else {
            DMatrix::zeros(parameters_gradient.nrows(), parameters_gradient.ncols())
        };

        self.v = Some(self.momentum * v + lr * parameters_gradient);
        parameters - (lr * parameters_gradient)
    }
}
