use crate::linalg::Scalar;
use crate::ops::{model::Model, Data};

use super::op_node::OpNodeTrait;

pub struct OpSubgraph<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>(pub Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>);

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpSubgraph<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new(op: Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>) -> Self {
        Self(op)
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>> Model
    for OpSubgraph<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn get_learnable_params_count(&self) -> usize {
        self.0.get_learnable_params_count()
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        self.0.load_learnable_params(params);
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        self.0.get_learnable_params()
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpSubgraph<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn forward_or_transform_inference(&mut self, input: DataIn) -> DataOut {
        self.0.forward_or_transform_inference(input)
    }

    fn forward_or_transform(
        &mut self,
        input: DataIn,
        reference: DataRefIn,
    ) -> (DataOut, DataRefOut) {
        self.0.forward_or_transform(input, reference)
    }

    fn backward_or_revert(
        &mut self,
        incoming_grad: DataOut,
        reference: DataRefOut,
    ) -> (DataIn, DataRefIn) {
        self.0.backward_or_revert(incoming_grad, reference)
    }

    fn revert_reference(&mut self, reference: DataRefOut) -> DataRefIn {
        self.0.revert_reference(reference)
    }
}

pub type OpGraph<'g, DataOut, DataRefOut> = OpSubgraph<'g, (), DataOut, (), DataRefOut>;

impl<'g, DataOut: Data<'g>, DataRefOut: Data<'g>> OpGraph<'g, DataOut, DataRefOut> {
    pub fn run_inference(&mut self) -> DataOut {
        self.0.forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        self.0.forward_or_transform((), ())
    }
}
