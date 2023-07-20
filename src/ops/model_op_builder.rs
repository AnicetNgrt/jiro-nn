use super::{Data, ModelOp, OpChain};

pub struct OpBuild<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    pub builder: Box<dyn Fn() -> Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>> + 'a>,
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuild<'a, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new_fn(
        builder: fn() -> Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
    ) -> Self {
        Self {
            builder: Box::new(builder),
        }
    }

    pub fn new_op_builder<OpB: OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut> + 'a>(
        builder: OpB,
    ) -> Self {
        Self {
            builder: Box::new(move || builder.build()),
        }
    }
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuild<'a, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(&self) -> Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>> {
        (self.builder)()
    }
}

pub trait OpBuilder<DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn build(&self) -> Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>;
}

pub struct OpBuilderChain<
    'a,
    DataIn: Data,
    DataMid: Data,
    DataOut: Data,
    DataRefIn: Data,
    DataRefMid: Data,
    DataRefOut: Data,
> {
    first_op: Box<dyn OpBuilder<DataIn, DataMid, DataRefIn, DataRefMid> + 'a>,
    second_op: Box<dyn OpBuilder<DataMid, DataOut, DataRefMid, DataRefOut> + 'a>,
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > OpBuilderChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn OpBuilder<DataIn, DataMid, DataRefIn, DataRefMid> + 'a>,
        second_op: Box<dyn OpBuilder<DataMid, DataOut, DataRefMid, DataRefOut> + 'a>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuilderChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn build(&self) -> Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>> {
        let first_op = self.first_op.build();
        let second_op = self.second_op.build();
        Box::new(OpChain::new(first_op, second_op))
    }
}

pub trait CombinatoryOpBuilder<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn push_and_chain<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<'a, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>;

    fn push_and_pack<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuild<'a, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed>;
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data, OpB>
    CombinatoryOpBuilder<'a, DataIn, DataOut, DataRefIn, DataRefOut> for OpB
where
    OpB: OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut> + 'a,
{
    fn push_and_chain<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<'a, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>
    {
        OpBuilderChain::new(Box::new(self), Box::new(op))
    }

    fn push_and_pack<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuild<'a, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed> {
        OpBuild::new_op_builder(self.push_and_chain(op))
    }
}
