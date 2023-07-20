use super::{combinatory_op::OriginOp, Data, ModelOp, OpChain};

pub struct ModelBuilderOrigin<D: Data, DataRef: Data> {
    _phantom: std::marker::PhantomData<(D, DataRef)>,
}

impl<D: Data, DataRef: Data> ModelBuilderOrigin<D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: Data, DataRef: Data> OpBuilder<D, D, DataRef, DataRef> for ModelBuilderOrigin<D, DataRef> {
    fn build(
        &self,
        sample_data: D,
        sample_ref: DataRef,
    ) -> (Box<dyn ModelOp<D, D, DataRef, DataRef>>, (D, DataRef)) {
        (Box::new(OriginOp::new()), (sample_data, sample_ref))
    }
}

pub struct OpBuild<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    pub builder: Box<
        dyn Fn(
                DataIn,
                DataRefIn,
            ) -> (
                Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
                (DataIn, DataRefIn),
            ) + 'a,
    >,
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuild<'a, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new_fn(
        builder: fn(
            DataIn,
            DataRefIn,
        ) -> (
            Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
            (DataIn, DataRefIn),
        ),
    ) -> Self {
        Self {
            builder: Box::new(builder),
        }
    }

    pub fn new_op_builder<OpB: OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut> + 'a>(
        builder: OpB,
    ) -> Self {
        Self {
            builder: Box::new(move |sample_data, sample_ref| {
                builder.build(sample_data, sample_ref)
            }),
        }
    }
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuild<'a, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(
        &self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
        (DataIn, DataRefIn),
    ) {
        (self.builder)(sample_data, sample_ref)
    }
}

pub trait OpBuilder<DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn build(
        &self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
        (DataIn, DataRefIn),
    );
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
    fn build(
        &self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
        (DataIn, DataRefIn),
    ) {
        let (mut first_op, (sample_data, sample_ref)) =
            self.first_op.build(sample_data, sample_ref);
        let (sample_data, sample_ref) = first_op.forward_or_transform(sample_data, sample_ref);
        let (second_op, (sample_data, sample_ref)) = self.second_op.build(sample_data, sample_ref);
        let data_and_ref = first_op.backward_or_revert(sample_data, sample_ref);
        (Box::new(OpChain::new(first_op, second_op)), data_and_ref)
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
