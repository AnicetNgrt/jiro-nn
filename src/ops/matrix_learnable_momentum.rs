use crate::{
    learning_rate::LearningRateScheduler,
    linalg::{Matrix, MatrixTrait, Scalar},
};

use super::{model::{impl_model_from_model_fields, Model}, optimizer::Optimizer};

pub struct MatrixLearnableMomentum {
    parameter: Matrix,
    learning_rate: Box<dyn LearningRateScheduler>,
    momentum: Scalar,
    v: Option<Matrix>,
}

impl MatrixLearnableMomentum {
    pub fn new(
        parameter: Matrix,
        learning_rate: Box<dyn LearningRateScheduler>,
        momentum: Scalar,
    ) -> Self {
        Self {
            parameter,
            learning_rate,
            momentum,
            v: None,
        }
    }
}

impl Optimizer<Matrix> for MatrixLearnableMomentum {
    fn step(&mut self, incoming_grad: &Matrix) {
        let lr = self.learning_rate.get_learning_rate();
        self.learning_rate.increment_step();

        if let None = &self.v {
            let (nrow, ncol) = incoming_grad.dim();
            self.v = Some(Matrix::zeros(nrow, ncol));
        };

        let v = self.v.as_ref().unwrap();

        let v = v
            .scalar_mul(self.momentum)
            .component_add(&incoming_grad.scalar_mul(lr));

        self.parameter = incoming_grad.component_sub(&v);
        self.v = Some(v);
    }

    fn get_param(&self) -> &Matrix {
        &self.parameter
    }
}

impl Model for MatrixLearnableMomentum {
    impl_model_from_model_fields!(parameter);
} 