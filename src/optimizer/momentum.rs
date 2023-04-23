use serde::{Deserialize, Serialize};

use crate::{learning_rate::{LearningRateSchedule, default_learning_rate}, linalg::{Matrix, MatrixTrait}};

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
    v: Option<Matrix>,
}

impl Momentum {
    pub fn new(learning_rate: LearningRateSchedule, momentum: f64) -> Self {
        Self {
            v: None,
            momentum,
            learning_rate,
        }
    }

    pub fn update_parameters(&mut self, epoch: usize, parameters: &Matrix, parameters_gradient: &Matrix) -> Matrix {
        let lr = self.learning_rate.get_learning_rate(epoch);

        let v = if let Some(v) = self.v.clone() {
            v
        } else {
            let (nrow, ncol) = parameters_gradient.dim();
            Matrix::zeros(nrow, ncol)
        };

        let v = v.scalar_mul(self.momentum).component_add(&parameters_gradient.scalar_mul(lr));
        
        let new_params = parameters.component_sub(&v);
        self.v = Some(v);
        new_params
    }
}
