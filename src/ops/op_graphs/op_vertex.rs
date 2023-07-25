use crate::linalg::Scalar;
use crate::ops::{
    model::{impl_model_from_model_fields, Model},
    Data,
};

use super::op_node::OpNodeTrait;

pub struct OpVertex<
    'g,
    DataIn: Data<'g>,
    DataMid: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefMid: Data<'g>,
    DataRefOut: Data<'g>,
> {
    first_op: Box<dyn OpNodeTrait<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
    second_op: Box<dyn OpNodeTrait<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > Model for OpVertex<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    impl_model_from_model_fields!(first_op, second_op);
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > OpNodeTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpVertex<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn forward_or_transform_inference(&mut self, input: DataIn) -> DataOut {
        let mid = self.first_op.forward_or_transform_inference(input);
        self.second_op.forward_or_transform_inference(mid)
    }

    fn forward_or_transform(
        &mut self,
        input: DataIn,
        reference: DataRefIn,
    ) -> (DataOut, DataRefOut) {
        let (mid, reference) = self.first_op.forward_or_transform(input, reference);
        self.second_op.forward_or_transform(mid, reference)
    }

    fn backward_or_revert(
        &mut self,
        output: DataOut,
        reference: DataRefOut,
    ) -> (DataIn, DataRefIn) {
        let (mid, reference) = self.second_op.backward_or_revert(output, reference);
        self.first_op.backward_or_revert(mid, reference)
    }

    fn revert_reference(&mut self, reference: DataRefOut) -> DataRefIn {
        self.first_op
            .revert_reference(self.second_op.revert_reference(reference))
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
    > OpVertex<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn OpNodeTrait<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
        second_op: Box<dyn OpNodeTrait<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}
