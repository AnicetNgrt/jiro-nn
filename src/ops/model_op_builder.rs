use super::{
    combinatory_op::{OpGraph, OriginOp},
    mapping::{InputMappingOp, ReferenceMappingOp},
    Data, ModelOp, OpChain,
};

pub struct OpGraphBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
> {
    pub builder: Option<
        Box<
            dyn FnOnce(
                    DataIn,
                    DataRefIn,
                ) -> (
                    Box<dyn ModelOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
                    (DataIn, DataRefIn),
                ) + 'g,
        >,
    >,
}

impl<'g, D: Data<'g> + Clone, DataRef: Data<'g> + Clone>
    OpGraphBuilder<'g, (), D, (), DataRef>
{
    pub fn data_as_entry_point(data: D, reference: DataRef) -> Self {
        let origin_op = OriginOp::<(), ()>::new();
        let chain_op = OpChain::new(
            Box::new(origin_op),
            Box::new(InputMappingOp::new(move |_| data.clone(), |_| ())),
        );
        let chain_op = OpChain::new(
            Box::new(chain_op),
            Box::new(ReferenceMappingOp::new(move |_| reference.clone(), |_| ())),
        );

        Self {
            builder: Some(Box::new(move |_, _| (Box::new(chain_op), ((), ())))),
        }
    }

    pub fn build_graph(&mut self) -> OpGraph<'g, D, DataRef> {
        match self.builder.take() {
            None => panic!("OpGraphBuilder::build() called twice."),
            Some(builder) => OpGraph((builder)((), ()).0),
        }
    }
}

impl<
        'g,
        DataIn: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefOut: Data<'g>,
    > OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new_fn(
        builder: Box<
            dyn FnOnce(
                    DataIn,
                    DataRefIn,
                ) -> (
                    Box<dyn ModelOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
                    (DataIn, DataRefIn),
                ) + 'g,
        >,
    ) -> Self {
        Self {
            builder: Some(builder),
        }
    }

    pub fn new_op_builder<
        OpB: OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
    >(
        mut builder: OpB,
    ) -> Self {
        Self {
            builder: Some(Box::new(move |sample_data, sample_ref| {
                builder.build(sample_data, sample_ref)
            })),
        }
    }
}

impl<
        'g,
        DataIn: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefOut: Data<'g>,
    > OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataIn, DataRefIn),
    ) {
        match self.builder.take() {
            None => panic!("OpGraphBuilder::build() called twice."),
            Some(builder) => (builder)(sample_data, sample_ref),
        }
    }
}

macro_rules! plug_builder_on_opbuild_data_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<
                'g,
                DataIn: Data<'g>,
                DataRefIn: Data<'g>,
                DataRefOut: Data<'g>,
            > OpGraphBuilder<'g, DataIn, $plug_type, DataRefIn, DataRefOut>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'g, DataIn, $out_type, DataRefIn, DataRefOut> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_data_out;

macro_rules! plug_builder_on_opbuild_reference_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<
                'g,
                DataIn: Data<'g>,
                DataOut: Data<'g>,
                DataRefIn: Data<'g>,
            > OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, $plug_type>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_reference_out;

macro_rules! plug_builder_on_opbuild_total_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>>
            OpGraphBuilder<'g, DataIn, $plug_type, DataRefIn, $plug_type>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'g, DataIn, $out_type, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_total_out;

pub trait OpSubgraphBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataIn, DataRefIn),
    );
}

pub struct OpBuilderChain<
    'g,
    DataIn: Data<'g>,
    DataMid: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefMid: Data<'g>,
    DataRefOut: Data<'g>,
> {
    first_op:
        Box<dyn OpSubgraphBuilder<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
    second_op:
        Box<dyn OpSubgraphBuilder<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > OpBuilderChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<
            dyn OpSubgraphBuilder<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g,
        >,
        second_op: Box<
            dyn OpSubgraphBuilder<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g,
        >,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuilderChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
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

pub trait CombinatoryOpBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>
{
    fn push_and_chain<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<
        'g,
        DataIn,
        DataOut,
        DataOutPushed,
        DataRefIn,
        DataRefOut,
        DataRefOutPushed,
    >;

    fn push_and_pack<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpGraphBuilder<'g, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed>;
}

impl<
        'g,
        DataIn: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefOut: Data<'g>,
        OpB,
    > CombinatoryOpBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> for OpB
where
    OpB: OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
{
    fn push_and_chain<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<
        'g,
        DataIn,
        DataOut,
        DataOutPushed,
        DataRefIn,
        DataRefOut,
        DataRefOutPushed,
    > {
        OpBuilderChain::new(Box::new(self), Box::new(op))
    }

    fn push_and_pack<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpGraphBuilder<'g, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed> {
        OpGraphBuilder::new_op_builder(self.push_and_chain(op))
    }
}
