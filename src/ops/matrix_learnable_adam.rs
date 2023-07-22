use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    learning_rate::{ConstantLearningRate, LearningRateScheduler},
    model::{impl_model_from_model_fields, Model},
    optimizer::{Optimizer, OptimizerBuilder},
};

pub struct MatrixLearnableAdam<'g> {
    parameter: Matrix,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
    m: Option<Matrix>, // first moment vector
    v: Option<Matrix>, // second moment vector
    learning_rate: Box<dyn LearningRateScheduler<'g> + 'g>,
}

impl<'g> MatrixLearnableAdam<'g> {
    pub fn new(
        parameter: Matrix,
        learning_rate: Box<dyn LearningRateScheduler<'g> + 'g>,
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

impl<'g> Optimizer<'g, Matrix> for MatrixLearnableAdam<'g> {
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

impl<'g> Model for MatrixLearnableAdam<'g> {
    impl_model_from_model_fields!(parameter);
}

pub struct MatrixLearnableAdamBuilder<'g, Parent: 'g> {
    learning_rate: Box<dyn LearningRateScheduler<'g> + 'g>,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
    parent_acceptor: Option<Box<dyn FnOnce(MatrixLearnableAdamBuilder<'g, Parent>) -> Parent + 'g>>,
}

impl<'g, Parent: 'g> MatrixLearnableAdamBuilder<'g, Parent> {
    pub fn new(
        parent_acceptor: Option<
            Box<dyn FnOnce(MatrixLearnableAdamBuilder<'g, Parent>) -> Parent + 'g>,
        >,
    ) -> Self {
        Self {
            learning_rate: Box::new(ConstantLearningRate::new(0.001)),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            parent_acceptor,
        }
    }

    pub fn with_beta1(mut self, beta1: Scalar) -> Self {
        self.beta1 = beta1;
        self
    }

    pub fn with_beta2(mut self, beta2: Scalar) -> Self {
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: Scalar) -> Self {
        self.epsilon = epsilon;
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

impl<'g, Parent: 'g> OptimizerBuilder<'g, Matrix> for MatrixLearnableAdamBuilder<'g, Parent> {
    fn build(&self, parameter: Matrix) -> Box<dyn Optimizer<'g, Matrix> + 'g> {
        Box::new(MatrixLearnableAdam::new(
            parameter,
            self.learning_rate.clone_box(),
            self.beta1,
            self.beta2,
            self.epsilon,
        ))
    }

    fn clone_box(&self) -> Box<dyn OptimizerBuilder<'g, Matrix> + 'g> {
        Box::new(MatrixLearnableAdamBuilder::<bool> {
            learning_rate: self.learning_rate.clone_box(),
            parent_acceptor: None,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
        })
    }
}
