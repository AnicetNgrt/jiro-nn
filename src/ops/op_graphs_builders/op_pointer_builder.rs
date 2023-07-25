use std::{rc::Rc, cell::RefCell};

use crate::ops::{
    op_graphs::{op_graph_shared::OpNodeShared, op_node::OpNodeTrait},
    Data,
};

use super::{op_graph_builder::OpGraphBuilder, op_node_builder::OpNodeBuilder, linkable_op_builder::LinkableOpBuilder};

pub struct OpPointerBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
> where
    DataOut::Meta: Clone,
    DataRefOut::Meta: Clone,
{
    builder: Rc<RefCell<Box<dyn OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>>>,
    shared_op: Rc<RefCell<Option<OpNodeShared<'g, DataIn, DataOut, DataRefIn, DataRefOut>>>>,
    meta_data_out: Rc<RefCell<Option<DataOut::Meta>>>,
    meta_ref_out: Rc<RefCell<Option<DataRefOut::Meta>>>,
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpPointerBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
where
    DataOut::Meta: Clone,
    DataRefOut::Meta: Clone,
{
    pub fn new<OpB: OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>(
        builder: OpB,
    ) -> Self {
        Self {
            builder: Rc::new(RefCell::new(Box::new(builder))),
            shared_op: Rc::new(RefCell::new(None)),
            meta_data_out: Rc::new(RefCell::new(None)),
            meta_ref_out: Rc::new(RefCell::new(None)),
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            builder: self.builder.clone(),
            shared_op: self.shared_op.clone(),
            meta_data_out: self.meta_data_out.clone(),
            meta_ref_out: self.meta_ref_out.clone(),
        }
    }

    pub fn get_pointer_to_op(&self) -> OpNodeShared<'g, DataIn, DataOut, DataRefIn, DataRefOut> {
        self.shared_op.borrow().as_ref().unwrap().clone()
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpNodeBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpPointerBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
where
    DataOut::Meta: Clone,
    DataRefOut::Meta: Clone,
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g>,
        (DataOut::Meta, DataRefOut::Meta),
    ) {
        let shared_op = if let Some(shared_op) = self.shared_op.borrow().as_ref() {
            Some(shared_op.clone())
        } else {
            None
        };
        match shared_op {
            Some(op) => (
                Box::new(op.clone()),
                (
                    self.meta_data_out.borrow().as_ref().unwrap().clone(),
                    self.meta_ref_out.borrow().as_ref().unwrap().clone(),
                ),
            ),
            None => {
                let (op, (meta_data_out, meta_ref_out)) = self.builder.borrow_mut().build(meta_data, meta_ref);
                let op = OpNodeShared::new(op);
                let mut shared_op = self.shared_op.borrow_mut();
                *shared_op = Some(op.clone());
                let mut _meta_data_out = self.meta_data_out.borrow_mut();
                *_meta_data_out = Some(meta_data_out.clone());
                let mut _meta_ref_out = self.meta_ref_out.borrow_mut();
                *_meta_ref_out = Some(meta_ref_out.clone());
                (Box::new(op), (meta_data_out, meta_ref_out))
            }
        }
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn pointer<PointerDataOut: Data<'g>, PointerDataRefOut: Data<'g>>(
        self,
        pointer: &OpPointerBuilder<'g, DataOut, PointerDataOut, DataRefOut, PointerDataRefOut>,
    ) -> OpGraphBuilder<'g, DataIn, PointerDataOut, DataRefIn, PointerDataRefOut>
    where
        PointerDataOut::Meta: Clone,
        PointerDataRefOut::Meta: Clone,
    {
        self.link_and_pack(pointer.clone())
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>
where
    DataOut::Meta: Clone,
    DataRefOut::Meta: Clone,
{
    pub fn make_pointer(self) -> OpPointerBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut> {
        OpPointerBuilder::new(self)
    }
}