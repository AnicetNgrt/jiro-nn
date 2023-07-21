use super::{
    combinatory_op::{OpGraph, OriginOp},
    mapping::{InputMappingOp, ReferenceMappingOp},
    Data, ModelOp, OpChain,
};

pub struct OpGraphBuilder<
    'opgraph,
    DataIn: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
> {
    pub builder: Option<
        Box<
            dyn FnOnce(
                    DataIn,
                    DataRefIn,
                ) -> (
                    Box<dyn ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph>,
                    (DataIn, DataRefIn),
                ) + 'opgraph,
        >,
    >,
}

impl<'opgraph, D: Data<'opgraph> + Clone, DataRef: Data<'opgraph> + Clone>
    OpGraphBuilder<'opgraph, (), D, (), DataRef>
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

    pub fn build_graph(&mut self) -> OpGraph<'opgraph, D, DataRef> {
        match self.builder.take() {
            None => panic!("OpGraphBuilder::build() called twice."),
            Some(builder) => OpGraph((builder)((), ()).0),
        }
    }
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > OpGraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new_fn(
        builder: Box<
            dyn FnOnce(
                    DataIn,
                    DataRefIn,
                ) -> (
                    Box<dyn ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph>,
                    (DataIn, DataRefIn),
                ) + 'opgraph,
        >,
    ) -> Self {
        Self {
            builder: Some(builder),
        }
    }

    pub fn new_op_builder<
        OpB: OpSubgraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph,
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
        'opgraph,
        DataIn: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > OpSubgraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpGraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph>,
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
                'opgraph,
                DataIn: Data<'opgraph>,
                DataRefIn: Data<'opgraph>,
                DataRefOut: Data<'opgraph>,
            > OpGraphBuilder<'opgraph, DataIn, $plug_type, DataRefIn, DataRefOut>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'opgraph, DataIn, $out_type, DataRefIn, DataRefOut> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_data_out;

macro_rules! plug_builder_on_opbuild_reference_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<
                'opgraph,
                DataIn: Data<'opgraph>,
                DataOut: Data<'opgraph>,
                DataRefIn: Data<'opgraph>,
            > OpGraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, $plug_type>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_reference_out;

macro_rules! plug_builder_on_opbuild_total_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'opgraph, DataIn: Data<'opgraph>, DataRefIn: Data<'opgraph>>
            OpGraphBuilder<'opgraph, DataIn, $plug_type, DataRefIn, $plug_type>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'opgraph, DataIn, $out_type, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_total_out;

pub trait OpSubgraphBuilder<
    'opgraph,
    DataIn: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph>,
        (DataIn, DataRefIn),
    );
}

pub struct OpBuilderChain<
    'opgraph,
    DataIn: Data<'opgraph>,
    DataMid: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefMid: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
> {
    first_op:
        Box<dyn OpSubgraphBuilder<'opgraph, DataIn, DataMid, DataRefIn, DataRefMid> + 'opgraph>,
    second_op:
        Box<dyn OpSubgraphBuilder<'opgraph, DataMid, DataOut, DataRefMid, DataRefOut> + 'opgraph>,
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataMid: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefMid: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > OpBuilderChain<'opgraph, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<
            dyn OpSubgraphBuilder<'opgraph, DataIn, DataMid, DataRefIn, DataRefMid> + 'opgraph,
        >,
        second_op: Box<
            dyn OpSubgraphBuilder<'opgraph, DataMid, DataOut, DataRefMid, DataRefOut> + 'opgraph,
        >,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataMid: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefMid: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > OpSubgraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuilderChain<'opgraph, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph>,
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
    'opgraph,
    DataIn: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
>
{
    fn push_and_chain<
        DataOutPushed: Data<'opgraph>,
        DataRefOutPushed: Data<'opgraph>,
        OpBuilderPushed: OpSubgraphBuilder<'opgraph, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'opgraph,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<
        'opgraph,
        DataIn,
        DataOut,
        DataOutPushed,
        DataRefIn,
        DataRefOut,
        DataRefOutPushed,
    >;

    fn push_and_pack<
        DataOutPushed: Data<'opgraph>,
        DataRefOutPushed: Data<'opgraph>,
        OpBuilderPushed: OpSubgraphBuilder<'opgraph, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'opgraph,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpGraphBuilder<'opgraph, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed>;
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
        OpB,
    > CombinatoryOpBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> for OpB
where
    OpB: OpSubgraphBuilder<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut> + 'opgraph,
{
    fn push_and_chain<
        DataOutPushed: Data<'opgraph>,
        DataRefOutPushed: Data<'opgraph>,
        OpBuilderPushed: OpSubgraphBuilder<'opgraph, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'opgraph,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<
        'opgraph,
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
        DataOutPushed: Data<'opgraph>,
        DataRefOutPushed: Data<'opgraph>,
        OpBuilderPushed: OpSubgraphBuilder<'opgraph, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed>
            + 'opgraph,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpGraphBuilder<'opgraph, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed> {
        OpGraphBuilder::new_op_builder(self.push_and_chain(op))
    }
}
