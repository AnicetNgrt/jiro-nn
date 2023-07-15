use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{Op, matrix_learnable::MatrixLearnable};

pub struct BatchedVecsDenseLayer {
    weights: Box<dyn MatrixLearnable>,
    biases: Box<dyn MatrixLearnable>,
    input: Option<Matrix>,
}

impl Op<Matrix, Matrix> for BatchedVecsDenseLayer {
    fn forward_inference(&self, input: &Matrix) -> Matrix {
        let weights = self.weights.get();
        let biases = self.biases.get();

        let res = weights.dot(&input)
            .component_add(&biases.dot(&Matrix::constant(1, input.dim().1, 1.0)));

        res
    }
    
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());
        let res = self.forward_inference(input);
        res
    }

    fn backward(&mut self, incoming_grad: &Matrix) -> Matrix {
        let input = self.input.as_ref().unwrap();

        let weights_gradient = &incoming_grad.dot(&input.transpose());

        let biases_gradient = incoming_grad.columns_sum();

        let weights = self.weights.get();
        let input_gradient = weights.transpose().dot(&incoming_grad);
        
        self.weights.backward(weights_gradient);
        self.biases.backward(&biases_gradient);

        input_gradient
    }

    fn get_learnable_params_count(&self) -> usize {
        self.weights.get_learnable_params_count() + self.biases.get_learnable_params_count()
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        let weights_count = self.weights.get_learnable_params_count();
        let (weights_params, biases_params) = params.split_at(weights_count);
        self.weights.load_learnable_params(weights_params.to_vec());
        self.biases.load_learnable_params(biases_params.to_vec());
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        let mut params = self.weights.get_learnable_params();
        params.extend(self.biases.get_learnable_params());
        params
    }
}