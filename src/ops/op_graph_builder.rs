use super::{
    combinatory_op::OriginOp,
    mapping::{input_mapping::InputMappingOp, reference_mapping::ReferenceMappingOp},
    op_graph::{OpGraph, OpGraphShared, OpGraphThreadShared, OpSubgraphTrait},
    Data, OpChain,
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
                    DataIn::Meta,
                    DataRefIn::Meta,
                ) -> (
                    Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
                    (DataOut::Meta, DataRefOut::Meta),
                ) + 'g,
        >,
    >,
}

impl<'g, D: Data<'g> + Clone, DataRef: Data<'g> + Clone> OpGraphBuilder<'g, (), D, (), DataRef> {
    pub fn from_data(data: D, reference: DataRef) -> Self {
        let origin_op = OriginOp::<(), ()>::new();
        let data1 = data.clone();
        let data2 = data.clone();
        let chain_op = OpChain::new(
            Box::new(origin_op),
            Box::new(InputMappingOp::new(
                move |_| data1.clone(),
                |_| (),
                move |_| data2.meta(),
            )),
        );
        let ref1 = reference.clone();
        let ref2 = reference.clone();
        let mut chain_op = OpChain::new(
            Box::new(chain_op),
            Box::new(ReferenceMappingOp::new(
                move |_| ref1.clone(),
                |_| (),
                move |_| ref2.meta(),
            )),
        );

        Self {
            builder: Some(Box::new(move |_, _| {
                let (data, reference) = chain_op.forward_or_transform((), ());
                (Box::new(chain_op), (data.meta(), reference.meta()))
            })),
        }
    }
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OpGraphBuilder<'g, (), D, (), DataRef> {
    // pub fn subgraph_as_entry_point<Op: OpSubgraphTrait<'g, (), D, (), DataRef> + 'g>(op: Op) -> Self {
    //     Self {
    //         builder: Some(Box::new(move |_, _| (Box::new(op), ((), ())))),
    //     }
    // }

    pub fn build_partial(
        &mut self,
    ) -> (
        Box<dyn OpSubgraphTrait<'g, (), D, (), DataRef> + 'g>,
        (D::Meta, DataRef::Meta),
    ) {
        match self.builder.take() {
            None => panic!("Building called twice."),
            Some(builder) => (builder)((), ()),
        }
    }

    pub fn build_graph(&mut self) -> OpGraph<'g, D, DataRef> {
        match self.builder.take() {
            None => panic!("Building called twice."),
            Some(builder) => OpGraph::new((builder)((), ()).0),
        }
    }

    pub fn build_graph_shared(&mut self) -> OpGraphShared<'g, D, DataRef> {
        match self.builder.take() {
            None => panic!("Building called twice."),
            Some(builder) => OpGraphShared::new((builder)((), ()).0),
        }
    }

    pub fn build_graph_thread_shared(&mut self) -> OpGraphThreadShared<'g, D, DataRef> {
        match self.builder.take() {
            None => panic!("Building called twice."),
            Some(builder) => OpGraphThreadShared::new((builder)((), ()).0),
        }
    }

    // pub fn checkpoint(mut self, cx: Sender<OpGraphShared<'g, D, DataRef>>) -> Self {
    //     let shared_subgraph = self.build_graph_shared();
    //     cx.send(shared_subgraph.clone())
    //         .expect("Failed to send checkpoint to receiver");
    //     Self {
    //         builder: Some(Box::new(move |_, _| (Box::new(shared_subgraph), ((), ())))),
    //     }
    // }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn from_fn(
        builder: Box<
            dyn FnOnce(
                    DataIn::Meta,
                    DataRefIn::Meta,
                ) -> (
                    Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
                    (DataOut::Meta, DataRefOut::Meta),
                ) + 'g,
        >,
    ) -> Self {
        Self {
            builder: Some(builder),
        }
    }

    pub fn from_op_builder<
        OpB: OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
    >(
        mut builder: OpB,
    ) -> Self {
        Self {
            builder: Some(Box::new(move |meta_data, meta_ref| {
                builder.build(meta_data, meta_ref)
            })),
        }
    }

    // pub fn from_op_and_data<
    //     Op: OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
    // >(
    //     op: Op,
    //     sample_data: DataIn,
    //     sample_ref: DataRefIn,
    // ) -> Self {
    //     Self {
    //         builder: Some(Box::new(move |_, _| {
                
    //             (Box::new(op), data_and_ref)
    //         })),
    //     }
    // }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        match self.builder.take() {
            None => panic!("OpGraphBuilder::build() called twice."),
            Some(builder) => (builder)(meta_data, meta_ref),
        }
    }
}

macro_rules! plug_builder_on_op_subgraph_builder_data_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
            OpGraphBuilder<'g, DataIn, $plug_type, DataRefIn, DataRefOut>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'g, DataIn, $out_type, DataRefIn, DataRefOut> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_op_subgraph_builder_data_out;

macro_rules! plug_builder_on_op_subgraph_builder_reference_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>>
            OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, $plug_type>
        {
            pub fn $plug_name(self) -> OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_op_subgraph_builder_reference_out;

macro_rules! plug_builder_on_op_subgraph_builder_total_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>>
            OpGraphBuilder<'g, DataIn, $plug_type, DataRefIn, $plug_type>
        {
            pub fn $plug_name(self) -> OpGraphBuilder<'g, DataIn, $out_type, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_op_subgraph_builder_total_out;

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
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    );
}

pub struct OpSubgraphBuilderChain<
    'g,
    DataIn: Data<'g>,
    DataMid: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefMid: Data<'g>,
    DataRefOut: Data<'g>,
> {
    first_op: Box<dyn OpSubgraphBuilder<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
    second_op: Box<dyn OpSubgraphBuilder<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > OpSubgraphBuilderChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn OpSubgraphBuilder<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
        second_op: Box<dyn OpSubgraphBuilder<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
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
    for OpSubgraphBuilderChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        let (first_op, (meta_data, meta_ref)) =
            self.first_op.build(meta_data, meta_ref);
        let (second_op, (meta_data, meta_ref)) = self.second_op.build(meta_data, meta_ref);
        (Box::new(OpChain::new(first_op, second_op)), (meta_data, meta_ref))
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
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpSubgraphBuilderChain<
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
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpGraphBuilder<'g, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed>;
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>, OpB>
    CombinatoryOpBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> for OpB
where
    OpB: OpSubgraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
{
    fn push_and_chain<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpSubgraphBuilderChain<
        'g,
        DataIn,
        DataOut,
        DataOutPushed,
        DataRefIn,
        DataRefOut,
        DataRefOutPushed,
    > {
        OpSubgraphBuilderChain::new(Box::new(self), Box::new(op))
    }

    fn push_and_pack<
        DataOutPushed: Data<'g>,
        DataRefOutPushed: Data<'g>,
        OpBuilderPushed: OpSubgraphBuilder<'g, DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'g,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpGraphBuilder<'g, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed> {
        OpGraphBuilder::from_op_builder(self.push_and_chain(op))
    }
}
