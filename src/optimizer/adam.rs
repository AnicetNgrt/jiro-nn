use serde::{Deserialize, Serialize};

use crate::{
    learning_rate::{default_learning_rate, LearningRateSchedule},
    linalg::{Matrix, MatrixTrait},
};

fn default_beta1() -> f64 {
    0.9
}

fn default_beta2() -> f64 {
    0.999
}

fn default_epsilon() -> f64 {
    1e-8
}

// https://arxiv.org/pdf/1412.6980.pdf
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adam {
    #[serde(default = "default_beta1")]
    beta1: f64,
    #[serde(default = "default_beta2")]
    beta2: f64,
    #[serde(default = "default_epsilon")]
    epsilon: f64,
    #[serde(default = "default_learning_rate")]
    learning_rate: LearningRateSchedule,
    #[serde(skip)]
    m: Option<Matrix>, // first moment vector
    #[serde(skip)]
    v: Option<Matrix>, // second moment vector
}

impl Adam {
    pub fn new(learning_rate: LearningRateSchedule, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            m: None,
            v: None,
            beta1,
            beta2,
            learning_rate,
            epsilon,
        }
    }

    pub fn default() -> Self {
        Self {
            v: None,
            m: None,
            beta1: default_beta1(),
            beta2: default_beta2(),
            learning_rate: default_learning_rate(),
            epsilon: default_epsilon(),
        }
    }

    pub fn update_parameters(
        &mut self,
        epoch: usize,
        parameters: &Matrix,
        parameters_gradient: &Matrix,
    ) -> Matrix {
        let alpha = self.learning_rate.get_learning_rate(epoch);

        let (nrow, ncol) = parameters_gradient.dim();

        if self.m.is_none() {
            self.m = Some(Matrix::zeros(nrow, ncol));
        }
        if self.v.is_none() {
            self.v = Some(Matrix::zeros(nrow, ncol));
        }
        let mut m = self.m.clone().unwrap();
        let mut v = self.v.clone().unwrap();

        let g = parameters_gradient;
        let g2 = parameters_gradient.component_mul(&parameters_gradient);

        m = (m.scalar_mul(self.beta1)).component_add(&g.scalar_mul(1.0 - self.beta1));
        v = (v.scalar_mul(self.beta2)).component_add(&g2.scalar_mul(1.0 - self.beta2));

        let m_bias_corrected = m.scalar_div(1.0 - self.beta1);
        let v_bias_corrected = v.scalar_div(1.0 - self.beta2);

        let v_bias_corrected = v_bias_corrected.map(f64::sqrt);

        parameters.component_sub(
            &(m_bias_corrected.scalar_mul(alpha))
                .component_div(&v_bias_corrected.scalar_add(self.epsilon)),
        )
    }
}
