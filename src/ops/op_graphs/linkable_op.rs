use crate::ops::Data;

use super::{op_node::OpNodeTrait, op_vertex::OpVertex};

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
