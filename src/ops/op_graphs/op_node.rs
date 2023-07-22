use crate::ops::{model::Model, Data};

pub trait OpNodeTrait<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
>: Model
{
    fn forward_or_transform_inference(&mut self, input: DataIn) -> DataOut;
    fn forward_or_transform(
        &mut self,
        input: DataIn,
        reference: DataRefIn,
    ) -> (DataOut, DataRefOut);
    fn backward_or_revert(
        &mut self,
        incoming_grad: DataOut,
        reference: DataRefOut,
    ) -> (DataIn, DataRefIn);
}

macro_rules! impl_op_node_for_learnable_op {
    ($d:ident, $dref:ident) => {
        fn forward_or_transform_inference(&mut self, input: $d) -> $d {
            self.forward_inference(input)
        }

        fn forward_or_transform(&mut self, input: $d, reference: $dref) -> ($d, $dref) {
            (self.forward(input), reference)
        }

        fn backward_or_revert(&mut self, output: $d, reference: $dref) -> ($d, $dref) {
            (self.backward(output), reference)
        }
    };
}

pub(crate) use impl_op_node_for_learnable_op;

macro_rules! impl_op_node_for_input_transformation_op {
    ($din:ident, $dout:ident, $dref:ident, $drefout:ident) => {
        fn forward_or_transform_inference(&mut self, input: $din) -> $dout {
            self.transform(input)
        }

        fn forward_or_transform(&mut self, input: $din, reference: $dref) -> ($dout, $drefout) {
            (self.transform(input), reference)
        }

        fn backward_or_revert(&mut self, output: $dout, reference: $drefout) -> ($din, $dref) {
            (self.revert(output), reference)
        }
    };
}

pub(crate) use impl_op_node_for_input_transformation_op;

macro_rules! impl_op_node_for_reference_transformation_op {
    ($din:ident, $dout:ident, $dref:ident, $drefout:ident) => {
        fn forward_or_transform_inference(&mut self, input: $din) -> $dout {
            input
        }

        fn forward_or_transform(&mut self, input: $din, reference: $dref) -> ($dout, $drefout) {
            (input, self.transform(reference))
        }

        fn backward_or_revert(&mut self, output: $dout, reference: $drefout) -> ($din, $dref) {
            (output, self.revert(reference))
        }
    };
}

pub(crate) use impl_op_node_for_reference_transformation_op;

macro_rules! impl_op_node_for_total_transformation_op {
    ($din:ident, $dout:ident, $dref:ident, $drefout:ident) => {
        fn forward_or_transform_inference(&mut self, input: $din) -> $dout {
            self.transform(input)
        }

        fn forward_or_transform(&mut self, input: $din, reference: $dref) -> ($dout, $drefout) {
            (self.transform(input), self.transform(reference))
        }

        fn backward_or_revert(&mut self, output: $dout, reference: $drefout) -> ($din, $dref) {
            (self.revert(output), self.revert(reference))
        }
    };
}

pub(crate) use impl_op_node_for_total_transformation_op;