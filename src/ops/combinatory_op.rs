use crate::linalg::Scalar;

use super::{
    model::{impl_model_no_params, Model},
    op_graphs::op_node::OpNodeTrait,
    op_graphs::op_vertex::OpVertex,
    Data,
};

pub struct OriginOp<'g, D: Data<'g>, DataRef: Data<'g>> {
    _phantom: std::marker::PhantomData<&'g (D, DataRef)>,
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OriginOp<'g, D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> Model for OriginOp<'g, D, DataRef> {
    impl_model_no_params!();
}

impl<'g, D: Data<'g>, DataRef: Data<'g>> OpNodeTrait<'g, D, D, DataRef, DataRef>
    for OriginOp<'g, D, DataRef>
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

pub trait LinkableOp<
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
        OpLinked: OpNodeTrait<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpLinked,
    ) -> OpVertex<'g, DataIn, DataOut, DataOutLinked, DataRefIn, DataRefOut, DataRefOutLinked>;
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>, MOp>
    LinkableOp<'g, DataIn, DataOut, DataRefIn, DataRefOut> for MOp
where
    MOp: OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut> + 'g,
{
    fn link<
        DataOutLinked: Data<'g>,
        DataRefOutLinked: Data<'g>,
        OpLinked: OpNodeTrait<'g, DataOut, DataOutLinked, DataRefOut, DataRefOutLinked> + 'g,
    >(
        self,
        op: OpLinked,
    ) -> OpVertex<'g, DataIn, DataOut, DataOutLinked, DataRefIn, DataRefOut, DataRefOutLinked> {
        OpVertex::new(Box::new(self), Box::new(op))
    }
}
