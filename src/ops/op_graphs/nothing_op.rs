use crate::linalg::Scalar;

use crate::ops::model::impl_model_no_params;
use crate::ops::{
    mappings::{input_mapping::InputMappingOp, reference_mapping::ReferenceMappingOp},
    model::Model,
    Data,
};

use super::{op_node::OpNodeTrait, op_vertex::OpVertex};

pub struct NothingOp<'g, D: Data<'g>, DataRef: Data<'g>> {
    _phantom: std::marker::PhantomData<&'g (D, DataRef)>,
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> NothingOp<'g, D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> Model for NothingOp<'g, D, DataRef> {
    impl_model_no_params!();
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OpNodeTrait<'g, D, D, DataRef, DataRef>
    for NothingOp<'g, D, DataRef>
{
    fn forward_or_transform_inference(&mut self, input: D) -> D {
        input
    }

    fn forward_or_transform(&mut self, input: D, reference: DataRef) -> (D, DataRef) {
        (input, reference)
    }

    fn backward_or_revert(&mut self, output: D, reference: DataRef) -> (D, DataRef) {
        (output, reference)
    }
}

pub fn op_node_from_data<'g, D: Data<'g> + Clone>(
    data: D,
) -> Box<dyn OpNodeTrait<'g, (), D, (), ()> + 'g> {
    let origin_op = NothingOp::<(), ()>::new();
    let data1 = data.clone();
    let data2 = data.clone();
    let vertex_op = OpVertex::new(
        Box::new(origin_op),
        Box::new(InputMappingOp::new(
            move |_| data1.clone(),
            |_| (),
            move |_| data2.meta(),
        )),
    );

    Box::new(vertex_op)
}

pub fn op_node_with_ref<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRef: Data<'g> + Clone,
>(
    op: Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, ()> + 'g>,
    reference: DataRef,
) -> Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRef> + 'g> {
    let ref1 = reference.clone();
    let ref2 = reference.clone();
    let vertex_op = OpVertex::new(
        op,
        Box::new(ReferenceMappingOp::new(
            move |_| ref1.clone(),
            |_| (),
            move |_| ref2.meta(),
        )),
    );

    Box::new(vertex_op)
}
