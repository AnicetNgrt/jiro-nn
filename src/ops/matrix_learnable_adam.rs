use crate::{
    learning_rate::LearningRateScheduler,
    linalg::{Matrix, MatrixTrait, Scalar},
};

use super::{
    model::{
        impl_model_from_model_fields,
        Model,
    },
    optimizer::Optimizer,
};

pub struct MatrixLearnableAdam {
    parameter: Matrix,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
    m: Option<Matrix>, // first moment vector
    v: Option<Matrix>, // second moment vector
    learning_rate: Box<dyn LearningRateScheduler>,
}

impl MatrixLearnableAdam {
    pub fn new(
        parameter: Matrix,
        learning_rate: Box<dyn LearningRateScheduler>,
        beta1: Scalar,
        beta2: Scalar,
        epsilon: Scalar,
    ) -> Self {
        Self {
            parameter,
            m: None,
            v: None,
            beta1,
            beta2,
            learning_rate,
            epsilon,
        }
    }
}

impl Optimizer<Matrix> for MatrixLearnableAdam {
    fn step(&mut self, incoming_grad: &Matrix) {
        let alpha = self.learning_rate.get_learning_rate();
        self.learning_rate.increment_step();

        let (nrow, ncol) = incoming_grad.dim();

        if self.m.is_none() {
            self.m = Some(Matrix::zeros(nrow, ncol));
        }
        if self.v.is_none() {
            self.v = Some(Matrix::zeros(nrow, ncol));
        }
        let m = self.m.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();

        let g = incoming_grad;
        let g2 = incoming_grad.component_mul(&incoming_grad);

        let m = &(m.scalar_mul(self.beta1)).component_add(&g.scalar_mul(1.0 - self.beta1));
        let v = &(v.scalar_mul(self.beta2)).component_add(&g2.scalar_mul(1.0 - self.beta2));

        let m_bias_corrected = m.scalar_div(1.0 - self.beta1);
        let v_bias_corrected = v.scalar_div(1.0 - self.beta2);

        let v_bias_corrected = v_bias_corrected.sqrt();

        self.m = Some(m.clone());
        self.v = Some(v.clone());
        self.parameter = self.parameter.component_sub(
            &(m_bias_corrected.scalar_mul(alpha))
                .component_div(&v_bias_corrected.scalar_add(self.epsilon)),
        );
    }

    fn get_param(&self) -> &Matrix {
        &self.parameter
    }
}

impl Model for MatrixLearnableAdam {
    impl_model_from_model_fields!(parameter);
} 