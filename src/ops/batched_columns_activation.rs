use crate::linalg::{Scalar, Matrix, MatrixTrait};

use super::{model::{Model, impl_model_no_params}, LearnableOp};

pub struct BatchedColumnsActivation {
    f: fn(&Matrix) -> Matrix,
    fp: fn(&Matrix) -> Matrix,
    data_in: Option<Matrix>,
}

impl BatchedColumnsActivation {
    pub fn new(
        activation: fn(&Matrix) -> Matrix,
        activation_prime: fn(&Matrix) -> Matrix,
    ) -> Self {
        Self {
            f: activation,
            fp: activation_prime,
            data_in: None,
        }
    }
}

impl Model for BatchedColumnsActivation {
    impl_model_no_params!();
}

impl LearnableOp<Matrix> for BatchedColumnsActivation {
    fn forward_inference(&mut self, input: Matrix) -> Matrix {
        (self.f)(&input)
    }

    fn forward(&mut self, input: Matrix) -> Matrix {
        let res = (self.f)(&input);
        self.data_in = Some(input);
        res
    }

    fn backward(&mut self, incoming_grad: Matrix) -> Matrix {
        let data_in = self.data_in.as_ref().expect("No input: You can't run the backward pass on an Activation layer without running the forward pass first");
        let f_prime_x = (self.fp)(data_in);
        incoming_grad.component_mul(&f_prime_x)
    }
}
