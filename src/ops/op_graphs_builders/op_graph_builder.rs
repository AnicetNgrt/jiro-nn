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

    pub fn build_subgraph(
        mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> OpSubgraph<'g, DataIn, DataOut, DataRefIn, DataRefOut> {
        OpSubgraph::new(self.build(meta_data, meta_ref).0)
    }
}

impl<'g, D: Data<'g> + Clone, DataRef: Data<'g> + Clone> OpGraphBuilder<'g, (), D, (), DataRef> {
    pub fn from_data_and_ref(data: D, reference: DataRef) -> Self {
        Self {
            builder: Some(Box::new((data, reference))),
        }
    }
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OpGraphBuilder<'g, (), D, (), DataRef> {
    pub fn build_graph(
        mut self,
    ) -> OpGraph<'g, D, DataRef> {
        OpGraph::new(self.build((), ()).0)
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

use crate::ops::{
    op_graphs::{op_graph::{OpSubgraph, OpGraph}, op_node::OpNodeTrait},
    Data,
};

use super::op_node_builder::OpNodeBuilder;
