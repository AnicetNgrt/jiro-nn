use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    learning_rate::{ConstantLearningRate, LearningRateScheduler},
    model::{impl_model_from_model_fields, Model},
    optimizer::{Optimizer, OptimizerBuilder},
};

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

pub struct MatrixLearnableSGDBuilder<Parent> {
    learning_rate: Box<dyn LearningRateScheduler>,
    parent_acceptor: Option<Box<dyn FnOnce(MatrixLearnableSGDBuilder<Parent>) -> Parent>>,
}

impl<Parent> MatrixLearnableSGDBuilder<Parent> {
    pub fn new(
        parent_acceptor: Option<Box<dyn FnOnce(MatrixLearnableSGDBuilder<Parent>) -> Parent>>,
    ) -> Self {
        Self {
            learning_rate: Box::new(ConstantLearningRate::new(0.001)),
            parent_acceptor,
        }
    }

    pub fn constant_learning_rate(mut self, learning_rate: Scalar) -> Self {
        self.learning_rate = Box::new(ConstantLearningRate::new(learning_rate));
        self
    }

    pub fn end(mut self) -> Parent {
        let acceptor = self
            .parent_acceptor
            .take()
            .expect("Can't .end() if there is no parent. .build() instead.");
        (acceptor)(self)
    }
}

impl<Parent> OptimizerBuilder<Matrix>
    for MatrixLearnableSGDBuilder<Parent>
{
    fn build(self, parameter: Matrix) -> Box<dyn Optimizer<Matrix>> {
        Box::new(MatrixLearnableSGD::new(parameter, self.learning_rate))
    }

    fn clone_box(&self) -> Box<dyn OptimizerBuilder<Matrix>> {
        Box::new(MatrixLearnableSGDBuilder::<bool> {
            learning_rate: self.learning_rate.clone_box(),
            parent_acceptor: None,
        })
    }
}

pub trait AcceptsMatrixLearnableSGDBuilder: Sized {
    fn accept(self, builder: MatrixLearnableSGDBuilder<Self>) -> Self;
}
