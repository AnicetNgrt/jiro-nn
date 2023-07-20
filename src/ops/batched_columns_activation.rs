use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    impl_model_op_for_learnable_op,
    model::{impl_model_no_params, Model},
    model_op_builder::OpBuilder,
    Data, LearnableOp, ModelOp,
};

pub struct BatchedColumnsActivation {
    f: fn(&Matrix) -> Matrix,
    fp: fn(&Matrix) -> Matrix,
    data_in: Option<Matrix>,
}

impl BatchedColumnsActivation {
    pub fn new(activation: fn(&Matrix) -> Matrix, activation_prime: fn(&Matrix) -> Matrix) -> Self {
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

impl<DataRef: Data> ModelOp<Matrix, Matrix, DataRef, DataRef> for BatchedColumnsActivation {
    impl_model_op_for_learnable_op!(Matrix, DataRef);
}

pub struct BatchedColumnsActivationBuilder {
    f: fn(&Matrix) -> Matrix,
    fp: fn(&Matrix) -> Matrix,
}

impl BatchedColumnsActivationBuilder {
    pub fn new(activation: fn(&Matrix) -> Matrix, activation_prime: fn(&Matrix) -> Matrix) -> Self {
        Self {
            f: activation,
            fp: activation_prime,
        }
    }
}

impl<DataRef: Data> OpBuilder<Matrix, Matrix, DataRef, DataRef>
    for BatchedColumnsActivationBuilder
{
    fn build(&self) -> Box<dyn ModelOp<Matrix, Matrix, DataRef, DataRef>> {
        Box::new(BatchedColumnsActivation::new(self.f, self.fp))
    }
}

// macro_rules! push_batch_columns_activations {
//     () => {
//         pub fn tanh<DataRefOut: Data>(
//             self,
//         ) -> Box<dyn OpBuilder<Matrix, Matrix, DataRefOut, DataRefOut>> {
//             Box::new(self.push(TanhBuilder))
//         }

//         pub fn batched_columns_activation<DataRefOut: Data>(
//             self,
//             activation: BatchedColumnsActivationBuilder,
//         ) -> Box<dyn OpBuilder<Matrix, Matrix, DataRefOut, DataRefOut>> {
//             Box::new(self.push(activation))
//         }
//     };
//     ($drefout:ident) => {
//         pub fn tanh(self) -> Box<dyn OpBuilder<Matrix, Matrix, $drefout, $drefout>> {
//             Box::new(self.push(TanhBuilder))
//         }

//         pub fn batched_columns_activation(
//             self,
//             activation: BatchedColumnsActivation,
//         ) -> Box<dyn OpBuilder<Matrix, Matrix, $drefout, $drefout>> {
//             Box::new(self.push(activation))
//         }
//     };
// }

// pub(crate) use push_batch_columns_activations;
