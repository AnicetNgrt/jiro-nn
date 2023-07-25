use std::sync::{Arc, Mutex};

use crate::{
    linalg::Scalar,
    ops::{model::Model, Data},
};

use super::op_node::OpNodeTrait;

pub struct OpNodeThreadShared<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>(pub Arc<Mutex<Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>>>);

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpNodeThreadShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new(op: Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>) -> Self {
        Self(Arc::new(Mutex::new(op)))
    }

    pub fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>> Model
    for OpNodeThreadShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn get_learnable_params_count(&self) -> usize {
        let op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.get_learnable_params_count()
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.load_learnable_params(params);
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        let op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.get_learnable_params()
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpNodeThreadShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn forward_or_transform_inference(&mut self, input: DataIn) -> DataOut {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.forward_or_transform_inference(input)
    }

    fn forward_or_transform(
        &mut self,
        input: DataIn,
        reference: DataRefIn,
    ) -> (DataOut, DataRefOut) {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.forward_or_transform(input, reference)
    }

    fn backward_or_revert(
        &mut self,
        incoming_grad: DataOut,
        reference: DataRefOut,
    ) -> (DataIn, DataRefIn) {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.backward_or_revert(incoming_grad, reference)
    }

    fn revert_reference(&mut self, reference: DataRefOut) -> DataRefIn {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.revert_reference(reference)
    }
}

pub type OpGraphThreadShared<'g, DataOut, DataRefOut> =
    OpNodeThreadShared<'g, (), DataOut, (), DataRefOut>;

impl<'g, DataOut: Data<'g>, DataRefOut: Data<'g>> OpGraphThreadShared<'g, DataOut, DataRefOut> {
    pub fn run_inference(&mut self) -> DataOut {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        let mut op = self
            .0
            .try_lock()
            .expect("Failed to lock mutex around thread safe Op Graph");
        op.forward_or_transform((), ())
    }
}
