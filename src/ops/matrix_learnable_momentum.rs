use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    learning_rate::{ConstantLearningRate, LearningRateScheduler},
    model::{impl_model_from_model_fields, Model},
    optimizer::{Optimizer, OptimizerBuilder},
};

pub struct MatrixLearnableMomentum<'opgraph> {
    parameter: Matrix,
    learning_rate: Box<dyn LearningRateScheduler<'opgraph> + 'opgraph>,
    momentum: Scalar,
    v: Option<Matrix>,
}

impl<'opgraph> MatrixLearnableMomentum<'opgraph> {
    pub fn new(
        parameter: Matrix,
        learning_rate: Box<dyn LearningRateScheduler<'opgraph> + 'opgraph>,
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

impl<'opgraph> Optimizer<'opgraph, Matrix> for MatrixLearnableMomentum<'opgraph> {
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

impl<'opgraph> Model for MatrixLearnableMomentum<'opgraph> {
    impl_model_from_model_fields!(parameter);
}

pub struct MatrixLearnableMomentumBuilder<'opgraph, Parent: 'opgraph> {
    learning_rate: Box<dyn LearningRateScheduler<'opgraph> + 'opgraph>,
    momentum: Scalar,
    parent_acceptor: Option<
        Box<dyn FnOnce(MatrixLearnableMomentumBuilder<'opgraph, Parent>) -> Parent + 'opgraph>,
    >,
}

impl<'opgraph, Parent: 'opgraph> MatrixLearnableMomentumBuilder<'opgraph, Parent> {
    pub fn new(
        parent_acceptor: Option<
            Box<dyn FnOnce(MatrixLearnableMomentumBuilder<'opgraph, Parent>) -> Parent + 'opgraph>,
        >,
    ) -> Self {
        Self {
            learning_rate: Box::new(ConstantLearningRate::new(0.001)),
            momentum: 0.9,
            parent_acceptor,
        }
    }

    pub fn with_momentum(mut self, momentum: Scalar) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_constant_learning_rate(mut self, learning_rate: Scalar) -> Self {
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

impl<'opgraph, Parent: 'opgraph> OptimizerBuilder<'opgraph, Matrix>
    for MatrixLearnableMomentumBuilder<'opgraph, Parent>
{
    fn build(&self, parameter: Matrix) -> Box<dyn Optimizer<'opgraph, Matrix> + 'opgraph> {
        Box::new(MatrixLearnableMomentum::new(
            parameter,
            self.learning_rate.clone_box(),
            self.momentum,
        ))
    }

    fn clone_box(&self) -> Box<dyn OptimizerBuilder<'opgraph, Matrix> + 'opgraph> {
        Box::new(MatrixLearnableMomentumBuilder::<bool> {
            learning_rate: self.learning_rate.clone_box(),
            parent_acceptor: None,
            momentum: self.momentum,
        })
    }
}
