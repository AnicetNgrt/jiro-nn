use crate::{
    learning_rate::LearningRateScheduler,
    linalg::{Matrix, MatrixTrait, Scalar},
};

use super::{model::{impl_model_from_model_fields, Model}, optimizer::Optimizer};

pub struct MatrixLearnableSGD {
    parameter: Matrix,
    learning_rate: Box<dyn LearningRateScheduler>,
}

impl MatrixLearnableSGD {
    pub fn new(parameter: Matrix, learning_rate: Box<dyn LearningRateScheduler>) -> Self {
        Self {
            parameter,
            learning_rate,
        }
    }
}

impl Optimizer<Matrix> for MatrixLearnableSGD {
    fn step(&mut self, incoming_grad: &Matrix) {
        let lr = self.learning_rate.get_learning_rate();
        self.parameter = self.parameter.component_sub(&incoming_grad.scalar_mul(lr));
        self.learning_rate.increment_step();
    }

    fn get_param(&self) -> &Matrix {
        &self.parameter
    }
}

impl Model for MatrixLearnableSGD {
    impl_model_from_model_fields!(parameter);
} 
