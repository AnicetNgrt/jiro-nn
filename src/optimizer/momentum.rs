use serde::{Deserialize, Serialize};

use crate::{learning_rate::{LearningRateSchedule, default_learning_rate}, linalg::{Matrix, MatrixTrait, Scalar}};

fn default_momentum() -> Scalar {
    0.9
}

// https://arxiv.org/pdf/1207.0580.pdf
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Momentum {
    #[serde(default="default_momentum")]
    momentum: Scalar,
    #[serde(default="default_learning_rate")]
    learning_rate: LearningRateSchedule,
    #[serde(skip)]
    v: Option<Matrix>,
}

impl Momentum {
    pub fn new(learning_rate: LearningRateSchedule, momentum: Scalar) -> Self {
        Self {
            v: None,
            momentum,
            learning_rate,
        }
    }

    pub fn default() -> Self {
        Self {
            v: None,
            momentum: default_momentum(),
            learning_rate: default_learning_rate(),
        }
    }

    pub fn update_parameters(&mut self, epoch: usize, parameters: &Matrix, parameters_gradient: &Matrix) -> Matrix {
        let lr = self.learning_rate.get_learning_rate(epoch);

        if let None = &self.v {
            let (nrow, ncol) = parameters_gradient.dim();
            self.v = Some(Matrix::zeros(nrow, ncol));
        };

        let v = self.v.as_ref().unwrap();

        let v = v.scalar_mul(self.momentum).component_add(&parameters_gradient.scalar_mul(lr));
        
        let new_params = parameters.component_sub(&v);
        self.v = Some(v);
        new_params
    }
}
