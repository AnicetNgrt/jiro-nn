use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::learning_rate::LearningRateSchedule;

// https://arxiv.org/pdf/1207.0580.pdf
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Momentum {
    momentum: f64,
    learning_rate: LearningRateSchedule,
    #[serde(skip)]
    cached_learning_rate: Option<f64>,
    #[serde(skip)]
    v: Option<DMatrix<f64>>,
}

impl Momentum {
    pub fn new(learning_rate: LearningRateSchedule, momentum: f64) -> Self {
        Self {
            v: None,
            momentum,
            learning_rate,
            cached_learning_rate: None,
        }
    }

    pub fn update_learning_rate(&mut self, epoch: usize) -> f64 {
        let lr = self.learning_rate.get_learning_rate(epoch);
        self.cached_learning_rate = Some(lr);
        lr
    }

    pub fn update_weights(
        &mut self,
        _epoch: usize,
        weights: &DMatrix<f64>,
        weights_gradient: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let learning_rate = self.cached_learning_rate.unwrap();

        let v = if let Some(v) = self.v.clone() {
            v
        } else {
            DMatrix::zeros(weights_gradient.nrows(), weights_gradient.ncols())
        };

        self.v = Some(self.momentum * v + learning_rate * weights_gradient);
        weights - (learning_rate * weights_gradient)
    }
}
