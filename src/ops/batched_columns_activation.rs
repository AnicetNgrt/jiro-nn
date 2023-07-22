use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    model::{impl_model_no_params, Model},
    op_graph_builder::{CombinatoryOpBuilder, OpGraphBuilder, OpNodeBuilder},
    op_graphs::op_node::{impl_op_node_for_learnable_op, LearnableOp, OpNodeTrait},
    Data,
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

impl<'g> LearnableOp<'g, Matrix> for BatchedColumnsActivation {
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

impl<'g, DataRef: Data<'g>> OpNodeTrait<'g, Matrix, Matrix, DataRef, DataRef>
    for BatchedColumnsActivation
{
    impl_op_node_for_learnable_op!(Matrix, DataRef);
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

impl<'g, DataRef: Data<'g>> OpNodeBuilder<'g, Matrix, Matrix, DataRef, DataRef>
    for BatchedColumnsActivationBuilder
{
    fn build(
        &mut self,
        meta_data: (usize, usize),
        meta_ref: DataRef::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, Matrix, Matrix, DataRef, DataRef> + 'g>,
        ((usize, usize), DataRef::Meta),
    ) {
        (
            Box::new(BatchedColumnsActivation::new(self.f, self.fp)),
            (meta_data, meta_ref),
        )
    }
}

impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut>
{
    pub fn custom_activation(
        self,
        activation: fn(&Matrix) -> Matrix,
        activation_prime: fn(&Matrix) -> Matrix,
    ) -> OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut> {
        let builder = BatchedColumnsActivationBuilder::new(activation, activation_prime);
        self.link_and_pack(builder)
    }
}
