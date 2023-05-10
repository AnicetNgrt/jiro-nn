use serde::{Deserialize, Serialize};

use crate::{
    learning_rate::{default_learning_rate, LearningRateSchedule},
    linalg::{Scalar}, vision::{Image, image::ImageTrait},
};

fn default_beta1() -> Scalar {
    0.9
}

fn default_beta2() -> Scalar {
    0.999
}

fn default_epsilon() -> Scalar {
    1e-8
}

// https://arxiv.org/pdf/1412.6980.pdf
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adam {
    #[serde(default = "default_beta1")]
    beta1: Scalar,
    #[serde(default = "default_beta2")]
    beta2: Scalar,
    #[serde(default = "default_epsilon")]
    epsilon: Scalar,
    #[serde(default = "default_learning_rate")]
    learning_rate: LearningRateSchedule,
    #[serde(skip)]
    m: Option<Image>, // first moment vector
    #[serde(skip)]
    v: Option<Image>, // second moment vector
}

impl Adam {
    pub fn new(learning_rate: LearningRateSchedule, beta1: Scalar, beta2: Scalar, epsilon: Scalar) -> Self {
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
        parameters: &Image,
        parameters_gradient: &Image,
    ) -> Image {
        let alpha = self.learning_rate.get_learning_rate(epoch);

        let (nrow, ncol, nchan) = parameters_gradient.image_dims();
        let n_sample = parameters_gradient.samples();

        if self.m.is_none() {
            self.m = Some(Image::zeros(nrow, ncol, nchan , n_sample));
        }
        if self.v.is_none() {
            self.v = Some(Image::zeros(nrow, ncol, nchan , n_sample));
        }
        let m = self.m.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();

        let g = parameters_gradient;
        let g2 = parameters_gradient.component_mul(&parameters_gradient);

        let m = &(m.scalar_mul(self.beta1)).component_add(&g.scalar_mul(1.0 - self.beta1));
        let v = &(v.scalar_mul(self.beta2)).component_add(&g2.scalar_mul(1.0 - self.beta2));
        
        let m_bias_corrected = m.scalar_div(1.0 - self.beta1);
        let v_bias_corrected = v.scalar_div(1.0 - self.beta2);
        
        let v_bias_corrected = v_bias_corrected.sqrt();
        
        self.m = Some(m.clone());
        self.v = Some(v.clone());
        parameters.component_sub(
            &(m_bias_corrected.scalar_mul(alpha))
                .component_div(&v_bias_corrected.scalar_add(self.epsilon)),
        )
    }
}
