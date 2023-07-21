use crate::linalg::Scalar;

use super::{
    model::{impl_model_no_params, Model},
    Data, ModelOp, OpChain,
};

pub struct OpGraph<'a, DataOut: Data, DataRefOut: Data>(pub Box<dyn ModelOp<(), DataOut, (), DataRefOut> + 'a>);

impl<'a, DataOut: Data, DataRefOut: Data> OpGraph<'a, DataOut, DataRefOut> {
    pub fn run_inference(&mut self) -> DataOut {
        self.0.forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        self.0.forward_or_transform((), ())
    } 
}

pub struct OriginOp<D: Data, DataRef: Data> {
    _phantom: std::marker::PhantomData<(D, DataRef)>,
}

impl<D: Data, DataRef: Data> OriginOp<D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: Data, DataRef: Data> Model for OriginOp<D, DataRef> {
    impl_model_no_params!();
}

impl<D: Data, DataRef: Data> ModelOp<D, D, DataRef, DataRef> for OriginOp<D, DataRef> {
    fn forward_or_transform_inference(&mut self, input: D) -> D {
        input
    }

    fn forward_or_transform(&mut self, input: D, reference: DataRef) -> (D, DataRef) {
        (input, reference)
    }

    fn backward_or_revert(&mut self, output: D, reference: DataRef) -> (D, DataRef) {
        (output, reference)
    }
}

pub trait CombinatoryOp<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn push<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpPushed: ModelOp<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpPushed,
    ) -> OpChain<'a, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>;
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data, MOp>
    CombinatoryOp<'a, DataIn, DataOut, DataRefIn, DataRefOut> for MOp
where
    MOp: ModelOp<DataIn, DataOut, DataRefIn, DataRefOut> + 'a,
{
    fn push<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpPushed: ModelOp<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpPushed,
    ) -> OpChain<'a, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed> {
        OpChain::new(Box::new(self), Box::new(op))
    }
}