use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::learning_rate::{default_learning_rate, LearningRateSchedule};

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
    m: Option<DMatrix<f64>>, // first moment vector
    #[serde(skip)]
    v: Option<DMatrix<f64>>, // second moment vector
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
        parameters: &DMatrix<f64>,
        parameters_gradient: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let alpha = self.learning_rate.get_learning_rate(epoch);

        if self.m.is_none() {
            self.m = Some(DMatrix::zeros(
                parameters_gradient.nrows(),
                parameters_gradient.ncols(),
            ));
        }
        if self.v.is_none() {
            self.v = Some(DMatrix::zeros(
                parameters_gradient.nrows(),
                parameters_gradient.ncols(),
            ));
        }
        let mut m = self.m.clone().unwrap();
        let mut v = self.v.clone().unwrap();

        let g = parameters_gradient;
        let g2 = parameters_gradient.component_mul(&parameters_gradient);

        m = m * self.beta1 + g * (1.0 - self.beta1);
        v = v * self.beta2 + g2 * (1.0 - self.beta2);

        let m_bias_corrected = m / (1.0 - self.beta1);
        let mut v_bias_corrected = v / (1.0 - self.beta2);

        v_bias_corrected.apply(|el| {
            *el = el.sqrt();
        });

        parameters
            - (alpha * m_bias_corrected).component_div(&v_bias_corrected.add_scalar(self.epsilon))
    }
}
