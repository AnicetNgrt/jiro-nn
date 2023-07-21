use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::linalg::Scalar;

use super::{
    model::Model,
    Data,
};

pub trait OpSubgraphTrait<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>: Model
{
    fn forward_or_transform_inference(&mut self, input: DataIn) -> DataOut;
    fn forward_or_transform(
        &mut self,
        input: DataIn,
        reference: DataRefIn,
    ) -> (DataOut, DataRefOut);
    fn backward_or_revert(
        &mut self,
        incoming_grad: DataOut,
        reference: DataRefOut,
    ) -> (DataIn, DataRefIn);
}

macro_rules! impl_op_subgraph_for_learnable_op {
    ($d:ident, $dref:ident) => {
        fn forward_or_transform_inference(&mut self, input: $d) -> $d {
            self.forward_inference(input)
        }

        fn forward_or_transform(&mut self, input: $d, reference: $dref) -> ($d, $dref) {
            (self.forward(input), reference)
        }

        fn backward_or_revert(&mut self, output: $d, reference: $dref) -> ($d, $dref) {
            (self.backward(output), reference)
        }
    };
}

pub(crate) use impl_op_subgraph_for_learnable_op;

macro_rules! impl_op_subgraph_for_input_transformation_op {
    ($din:ident, $dout:ident, $dref:ident, $drefout:ident) => {
        fn forward_or_transform_inference(&mut self, input: $din) -> $dout {
            self.transform(input)
        }

        fn forward_or_transform(&mut self, input: $din, reference: $dref) -> ($dout, $drefout) {
            (self.transform(input), reference)
        }

        fn backward_or_revert(&mut self, output: $dout, reference: $drefout) -> ($din, $dref) {
            (self.revert(output), reference)
        }
    };
}

pub(crate) use impl_op_subgraph_for_input_transformation_op;

macro_rules! impl_op_subgraph_for_reference_transformation_op {
    ($din:ident, $dout:ident, $dref:ident, $drefout:ident) => {
        fn forward_or_transform_inference(&mut self, input: $din) -> $dout {
            input
        }

        fn forward_or_transform(&mut self, input: $din, reference: $dref) -> ($dout, $drefout) {
            (input, self.transform(reference))
        }

        fn backward_or_revert(&mut self, output: $dout, reference: $drefout) -> ($din, $dref) {
            (output, self.revert(reference))
        }
    };
}

pub(crate) use impl_op_subgraph_for_reference_transformation_op;

macro_rules! impl_op_subgraph_for_total_transformation_op {
    ($din:ident, $dout:ident, $dref:ident, $drefout:ident) => {
        fn forward_or_transform_inference(&mut self, input: $din) -> $dout {
            self.transform(input)
        }

        fn forward_or_transform(&mut self, input: $din, reference: $dref) -> ($dout, $drefout) {
            (self.transform(input), self.transform(reference))
        }

        fn backward_or_revert(&mut self, output: $dout, reference: $drefout) -> ($din, $dref) {
            (self.revert(output), self.revert(reference))
        }
    };
}

pub(crate) use impl_op_subgraph_for_total_transformation_op;

pub type OpGraphThreadShared<'g, DataOut, DataRefOut> =
    OpSubgraphThreadShared<'g, (), DataOut, (), DataRefOut>;

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

#[derive(Clone)]
pub struct OpSubgraphThreadShared<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>(pub Arc<Mutex<Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>>>);

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpSubgraphThreadShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new(op: Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>) -> Self {
        Self(Arc::new(Mutex::new(op)))
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>> Model
    for OpSubgraphThreadShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
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
    OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpSubgraphThreadShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
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
}

pub type OpGraphShared<'g, DataOut, DataRefOut> =
    OpSubgraphShared<'g, (), DataOut, (), DataRefOut>;

impl<'g, DataOut: Data<'g>, DataRefOut: Data<'g>> OpGraphShared<'g, DataOut, DataRefOut> {
    pub fn run_inference(&mut self) -> DataOut {
        self.0.borrow_mut().forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        self.0.borrow_mut().forward_or_transform((), ())
    }
}

#[derive(Clone)]
pub struct OpSubgraphShared<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>(pub Rc<RefCell<Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>>>);

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpSubgraphShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new(op: Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>) -> Self {
        Self(Rc::new(RefCell::new(op)))
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>> Model
    for OpSubgraphShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn get_learnable_params_count(&self) -> usize {
        self.0.borrow().get_learnable_params_count()
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        self.0.borrow_mut().load_learnable_params(params);
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        self.0.borrow().get_learnable_params()
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpSubgraphShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn forward_or_transform_inference(&mut self, input: DataIn) -> DataOut {
        self.0.borrow_mut().forward_or_transform_inference(input)
    }

    fn forward_or_transform(
        &mut self,
        input: DataIn,
        reference: DataRefIn,
    ) -> (DataOut, DataRefOut) {
        self.0.borrow_mut().forward_or_transform(input, reference)
    }

    fn backward_or_revert(
        &mut self,
        incoming_grad: DataOut,
        reference: DataRefOut,
    ) -> (DataIn, DataRefIn) {
        self.0.borrow_mut().backward_or_revert(incoming_grad, reference)
    }
}

pub type OpGraph<'g, DataOut, DataRefOut> =
    OpSubgraph<'g, (), DataOut, (), DataRefOut>;

impl<'g, DataOut: Data<'g>, DataRefOut: Data<'g>> OpGraph<'g, DataOut, DataRefOut> {
    pub fn run_inference(&mut self) -> DataOut {
        self.0.forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        self.0.forward_or_transform((), ())
    }
}

pub struct OpSubgraph<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>(pub Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>);

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpSubgraph<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new(op: Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>) -> Self {
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
    OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
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
}