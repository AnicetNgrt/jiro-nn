use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    impl_model_op_for_learnable_op,
    matrix_learnable_adam::MatrixLearnableAdamBuilder,
    model::{impl_model_from_model_fields, Model},
    model_op_builder::OpBuild,
    optimizer::{Optimizer, OptimizerBuilder},
    Data, LearnableOp, ModelOp, matrix_learnable_sgd::MatrixLearnableSGDBuilder,
};

pub struct BatchedColumnsDenseLayer {
    weights_optimizer: Box<dyn Optimizer<Matrix>>,
    biases_optimizer: Box<dyn Optimizer<Matrix>>,
    input: Option<Matrix>,
}

impl BatchedColumnsDenseLayer {
    pub fn new(
        weights_optimizer: Box<dyn Optimizer<Matrix>>,
        biases_optimizer: Box<dyn Optimizer<Matrix>>,
    ) -> Self {
        Self {
            weights_optimizer,
            biases_optimizer,
            input: None,
        }
    }
}

impl LearnableOp<Matrix> for BatchedColumnsDenseLayer {
    fn forward_inference(&mut self, input: Matrix) -> Matrix {
        let weights = self.weights_optimizer.get_param();
        let biases = self.biases_optimizer.get_param();

        let res = weights
            .dot(&input)
            .component_add(&biases.dot(&Matrix::constant(1, input.dim().1, 1.0)));

        res
    }

    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = Some(input.clone());
        let res = self.forward_inference(input);
        res
    }

    fn backward(&mut self, incoming_grad: Matrix) -> Matrix {
        let input = self.input.as_ref().unwrap();

        let weights_gradient = &incoming_grad.dot(&input.transpose());

        let biases_gradient = incoming_grad.columns_sum();

        let weights = self.weights_optimizer.get_param();
        let input_gradient = weights.transpose().dot(&incoming_grad);

        self.weights_optimizer.step(weights_gradient);
        self.biases_optimizer.step(&biases_gradient);

        input_gradient
    }
}

impl Model for BatchedColumnsDenseLayer {
    impl_model_from_model_fields!(weights_optimizer, biases_optimizer);
}

impl ModelOp<Matrix, Matrix, Matrix, Matrix> for BatchedColumnsDenseLayer {
    impl_model_op_for_learnable_op!(Matrix, Matrix);
}

pub struct BatchedColumnsDenseLayerBuilder {
    weights_optimizer: Box<dyn OptimizerBuilder<Matrix>>,
    biases_optimizer: Box<dyn OptimizerBuilder<Matrix>>,
}

impl BatchedColumnsDenseLayerBuilder {
    pub fn new() -> Self {
        Self {
            weights_optimizer: Box::new(MatrixLearnableAdamBuilder::<Self>::new(None)),
            biases_optimizer: Box::new(MatrixLearnableAdamBuilder::<Self>::new(None)),
        }
    }

    pub fn with_adam_optimized_weights(mut self) -> MatrixLearnableAdamBuilder<Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_adam_optimized_biases(mut self) -> MatrixLearnableAdamBuilder<Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_adam_optimized(mut self) -> MatrixLearnableAdamBuilder<Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_sgd_optimized_weights(mut self) -> MatrixLearnableSGDBuilder<Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_sgd_optimized_biases(mut self) -> MatrixLearnableSGDBuilder<Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_sgd_optimized(mut self) -> MatrixLearnableSGDBuilder<Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }
}

impl<'a, DataIn: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuild<'a, DataIn, Matrix, DataRefIn, DataRefOut>
{
    // pub fn dense(self) -> OpBuild<'a, DataIn, Matrix, DataRefIn, DataRefOut> {
    //     self.push_and_pack()
    // }
}
