use crate::linalg::Scalar;

use super::{
    model::{impl_model_no_params, Model},
    Data, OpSubgraph, OpChain,
};

pub struct OpGraph<'g, DataOut: Data<'g>, DataRefOut: Data<'g>>(
    pub Box<dyn OpSubgraph<'g, (), DataOut, (), DataRefOut> + 'g>,
);

impl<'g, DataOut: Data<'g>, DataRefOut: Data<'g>>
    OpGraph<'g, DataOut, DataRefOut>
{
    pub fn run_inference(&mut self) -> DataOut {
        self.0.forward_or_transform_inference(())
    }

    pub fn run(&mut self) -> (DataOut, DataRefOut) {
        self.0.forward_or_transform((), ())
    }
}

pub struct OriginOp<'g, D: Data<'g>, DataRef: Data<'g>> {
    _phantom: std::marker::PhantomData<&'g (D, DataRef)>,
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OriginOp<'g, D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> Model
    for OriginOp<'g, D, DataRef>
{
    impl_model_no_params!();
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OpSubgraph<'g, D, D, DataRef, DataRef>
    for OriginOp<'g, D, DataRef>
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
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>
{
    fn push<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpPushed: OpSubgraph<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'g,
    >(
        self,
        op: OpPushed,
    ) -> OpChain<'g, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>;
}

impl<
        'g,
        DataIn: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefOut: Data<'g>,
        MOp,
    > CombinatoryOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> for MOp
where
    MOp: OpSubgraph<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
{
    fn push<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpPushed: OpSubgraph<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'g,
    >(
        self,
        op: OpPushed,
    ) -> OpChain<'g, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>
    {
        OpChain::new(Box::new(self), Box::new(op))
    }
}
