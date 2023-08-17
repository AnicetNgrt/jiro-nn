use crate::ops::Data;

use super::{
    op_graph_builder::OpGraphBuilder, op_node_builder::OpNodeBuilder,
    op_vertex_builder::OpVertexBuilder,
};

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
    ) -> OpVertexBuilder<'g, DataIn, DataOut, DataOutLinked, DataRefIn, DataRefOut, DataRefOutLinked>;

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
    ) -> OpVertexBuilder<'g, DataIn, DataOut, DataOutLinked, DataRefIn, DataRefOut, DataRefOutLinked>
    {
        OpVertexBuilder::new(Box::new(self), Box::new(op))
    }

    fn link_and_pack<
        DataOutLinked: Data<'g>,
        DataRefOutLinked: Data<'g>,
        OpBuilderLinked: OpNodeBuilder<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpBuilderLinked,
    ) -> OpGraphBuilder<'g, DataIn, DataOutLinked, DataRefIn, DataRefOutLinked> {
        OpGraphBuilder::from_builder(self.link(op))
    }
}
