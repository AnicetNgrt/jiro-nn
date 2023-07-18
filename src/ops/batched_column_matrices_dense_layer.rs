use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    model::{impl_model_from_model_fields, Model},
    optimizer::Optimizer,
    LearnableOp, ModelOp, impl_model_op_for_learnable_op,
};

pub struct BatchedColumnMatricesDenseLayer {
    weights_optimizer: Box<dyn Optimizer<Matrix>>,
    biases_optimizer: Box<dyn Optimizer<Matrix>>,
    input: Option<Matrix>,
}

impl BatchedColumnMatricesDenseLayer {
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

impl LearnableOp<Matrix> for BatchedColumnMatricesDenseLayer {
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

impl Model for BatchedColumnMatricesDenseLayer {
    impl_model_from_model_fields!(
        weights_optimizer,
        biases_optimizer
    );
}

impl ModelOp<Matrix, Matrix, Matrix, Matrix> for BatchedColumnMatricesDenseLayer {
    impl_model_op_for_learnable_op!(Matrix, Matrix);
}