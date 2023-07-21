use crate::linalg::Scalar;

use super::{
    model::{impl_model_no_params, Model},
    Data, ModelOp, OpChain,
};

pub struct OpGraph<'opgraph, DataOut: Data<'opgraph>, DataRefOut: Data<'opgraph>>(
    pub Box<dyn ModelOp<'opgraph, (), DataOut, (), DataRefOut> + 'opgraph>,
);

impl<'opgraph, DataOut: Data<'opgraph>, DataRefOut: Data<'opgraph>>
    OpGraph<'opgraph, DataOut, DataRefOut>
{
    pub fn run_inference(&mut self) -> DataOut {
        self.0.forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        self.0.forward_or_transform((), ())
    }
}

pub struct OriginOp<'opgraph, D: Data<'opgraph>, DataRef: Data<'opgraph>> {
    _phantom: std::marker::PhantomData<&'opgraph (D, DataRef)>,
}

impl<'opgraph, D: Data<'opgraph>, DataRef: Data<'opgraph>> OriginOp<'opgraph, D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'opgraph, D: Data<'opgraph>, DataRef: Data<'opgraph>> Model
    for OriginOp<'opgraph, D, DataRef>
{
    impl_model_no_params!();
}

impl<'opgraph, D: Data<'opgraph>, DataRef: Data<'opgraph>> ModelOp<'opgraph, D, D, DataRef, DataRef>
    for OriginOp<'opgraph, D, DataRef>
{
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

pub trait CombinatoryOp<
    'opgraph,
    DataIn: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
>
{
    fn push<
        DataOutPushed: Data<'opgraph>,
        DataRefOutPushed: Data<'opgraph>,
        OpPushed: ModelOp<'opgraph, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'opgraph,
    >(
        self,
        op: OpPushed,
    ) -> OpChain<'opgraph, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>;
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
        MOp,
    > CombinatoryOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> for MOp
where
    MOp: ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph,
{
    fn push<
        DataOutPushed: Data<'opgraph>,
        DataRefOutPushed: Data<'opgraph>,
        OpPushed: ModelOp<'opgraph, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'opgraph,
    >(
        self,
        op: OpPushed,
    ) -> OpChain<'opgraph, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>
    {
        OpChain::new(Box::new(self), Box::new(op))
    }
}
