use super::{
    op_graphs::{op_node::OpNodeTrait, op_vertex::OpVertex},
    Data,
};

pub struct OpGraphBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
> {
    builder: Option<Box<dyn OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>>,
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn from_op_node_builder<
        OpB: OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
    >(
        builder: OpB,
    ) -> Self {
        Self {
            builder: Some(Box::new(builder)),
        }
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        match self.builder.take() {
            None => panic!("Building called twice."),
            Some(mut builder) => builder.build(meta_data, meta_ref),
        }
    }
}


macro_rules! plug_builder_on_op_node_builder_data_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
            OpGraphBuilder<'g, DataIn, $plug_type, DataRefIn, DataRefOut>
        {
            pub fn $plug_name(
                self,
            ) -> OpGraphBuilder<'g, DataIn, $out_type, DataRefIn, DataRefOut> {
                self.link_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_op_node_builder_data_out;

macro_rules! plug_builder_on_op_node_builder_reference_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>>
            OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, $plug_type>
        {
            pub fn $plug_name(self) -> OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, $out_type> {
                self.link_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_op_node_builder_reference_out;

macro_rules! plug_builder_on_op_node_builder_total_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>>
            OpGraphBuilder<'g, DataIn, $plug_type, DataRefIn, $plug_type>
        {
            pub fn $plug_name(self) -> OpGraphBuilder<'g, DataIn, $out_type, DataRefIn, $out_type> {
                self.link_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_op_node_builder_total_out;

pub trait OpNodeBuilder<
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
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    );
}

pub struct OpNodeChainBuilder<
    'g,
    DataIn: Data<'g>,
    DataMid: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefMid: Data<'g>,
    DataRefOut: Data<'g>,
> {
    first_op: Box<dyn OpNodeBuilder<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
    second_op: Box<dyn OpNodeBuilder<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > OpNodeChainBuilder<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn OpNodeBuilder<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
        second_op: Box<dyn OpNodeBuilder<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
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
    > OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpNodeChainBuilder<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        let (first_op, (meta_data, meta_ref)) = self.first_op.build(meta_data, meta_ref);
        let (second_op, (meta_data, meta_ref)) = self.second_op.build(meta_data, meta_ref);
        (
            Box::new(OpVertex::new(first_op, second_op)),
            (meta_data, meta_ref),
        )
    }
}

pub trait LinkableOpBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>
{
    fn link<
        DataOutLinked: Data<'g>,
        DataRefOutLinked: Data<'g>,
        OpBuilderLinked: OpNodeBuilder<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpBuilderLinked,
    ) -> OpNodeChainBuilder<
        'g,
        DataIn,
        DataOut,
        DataOutLinked,
        DataRefIn,
        DataRefOut,
        DataRefOutLinked,
    >;

    fn link_and_pack<
        DataOutLinked: Data<'g>,
        DataRefOutLinked: Data<'g>,
        OpBuilderLinked: OpNodeBuilder<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpBuilderLinked,
    ) -> OpGraphBuilder<'g, DataIn, DataOutLinked, DataRefIn, DataRefOutLinked>;
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>, OpB>
    LinkableOpBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> for OpB
where
    OpB: OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
{
    fn link<
        DataOutLinked: Data<'g>,
        DataRefOutLinked: Data<'g>,
        OpBuilderLinked: OpNodeBuilder<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpBuilderLinked,
    ) -> OpNodeChainBuilder<
        'g,
        DataIn,
        DataOut,
        DataOutLinked,
        DataRefIn,
        DataRefOut,
        DataRefOutLinked,
    > {
        OpNodeChainBuilder::new(Box::new(self), Box::new(op))
    }

    fn link_and_pack<
        DataOutLinked: Data<'g>,
        DataRefOutLinked: Data<'g>,
        OpBuilderLinked: OpNodeBuilder<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpBuilderLinked,
    ) -> OpGraphBuilder<'g, DataIn, DataOutLinked, DataRefIn, DataRefOutLinked> {
        OpGraphBuilder::from_op_node_builder(self.link(op))
    }
}