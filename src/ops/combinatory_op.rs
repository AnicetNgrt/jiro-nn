use super::{Data, ModelOp, OpChain};

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
