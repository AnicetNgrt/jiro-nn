use crate::ops::{
    op_graphs::{
        op_node::OpNodeTrait,
        origin_op::{op_node_from_data, op_node_with_ref},
    },
    Data,
};

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

impl<
        'g,
        DataIn: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefOut: Data<'g>,
        F: FnOnce(
            DataIn::Meta,
            DataRefIn::Meta,
        ) -> (
            Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
            (DataOut::Meta, DataRefOut::Meta),
        ),
    > OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> for Option<F>
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        match self.take() {
            None => panic!("Building called twice."),
            Some(f) => f(meta_data, meta_ref),
        }
    }
}

impl<
        'g,
        DataIn: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefOut: Data<'g>,
        F: FnMut(
            DataIn::Meta,
            DataRefIn::Meta,
        ) -> (
            Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
            (DataOut::Meta, DataRefOut::Meta),
        ),
    > OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> for F
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        (self)(meta_data, meta_ref)
    }
}

impl<'g, DataOut: Data<'g> + Clone, DataRefOut: Data<'g> + Clone>
    OpNodeBuilder<'g, (), DataOut, (), DataRefOut> for (DataOut, DataRefOut)
{
    fn build(
        &mut self,
        _meta_data: (),
        _meta_ref: (),
    ) -> (
        Box<dyn OpNodeTrait<'g, (), DataOut, (), DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        let op = op_node_from_data(self.0.clone());
        let op = op_node_with_ref(op, self.1.clone());
        (op, (self.0.meta(), self.1.meta()))
    }
}
