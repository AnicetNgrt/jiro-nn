use crate::ops::{
    op_graphs::{
        nothing_op::NothingOp,
        op_graph::{OpGraph, OpSubgraph},
        op_node::OpNodeTrait,
    },
    Data,
};

use super::{linkable_op_builder::LinkableOpBuilder, op_node_builder::OpNodeBuilder};

pub struct OpGraphBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
> {
    builder: Option<Box<dyn OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>>,
}

pub type OpGraphBuilderRoot<'g, D, DataRef> = OpGraphBuilder<'g, (), D, (), DataRef>;
pub type OpGraphBuilderEntry<'g, D, DataRef> = OpGraphBuilder<'g, D, D, DataRef, DataRef>;

pub fn graph_root<'g, D: Data<'g> + Clone, DataRef: Data<'g> + Clone>(data: D, reference: DataRef) -> OpGraphBuilderRoot<'g, D, DataRef> {
    OpGraphBuilderRoot::from_data(data, reference)
}

pub fn graph<'g, D: Data<'g>, DataRef: Data<'g>>() -> OpGraphBuilderEntry<'g, D, DataRef> {
    OpGraphBuilderEntry::new()
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new() -> OpGraphBuilder<'g, DataIn, DataIn, DataRefIn, DataRefIn> {
        let f = move |meta_data: DataIn::Meta,
                      meta_ref: DataRefIn::Meta|
              -> (
            Box<dyn OpNodeTrait<'g, DataIn, DataIn, DataRefIn, DataRefIn> + 'g>,
            (DataIn::Meta, DataRefIn::Meta),
        ) {
            let op = NothingOp::<DataIn, DataRefIn>::new();
            (Box::new(op), (meta_data, meta_ref))
        };

        OpGraphBuilder {
            builder: Some(Box::new(f)),
        }
    }

    pub fn from_builder<
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

    pub fn custom_node<
        BuilderDataOut: Data<'g>,
        BuilderDataRefOut: Data<'g>,
        OpB: OpNodeBuilder<'g, DataOut, BuilderDataOut, DataRefOut, BuilderDataRefOut> + 'g,
    >(
        self,
        builder: OpB,
    ) -> OpGraphBuilder<'g, DataIn, BuilderDataOut, DataRefIn, BuilderDataRefOut> {
        self.link_and_pack(builder)
    }
}

impl<'g, D: Data<'g> + Clone, DataRef: Data<'g> + Clone> OpGraphBuilder<'g, (), D, (), DataRef> {
    pub fn from_data(data: D, reference: DataRef) -> Self {
        Self {
            builder: Some(Box::new((data, reference))),
        }
    }
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OpGraphBuilder<'g, (), D, (), DataRef> {
    pub fn build_graph(mut self) -> OpGraph<'g, D, DataRef> {
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
