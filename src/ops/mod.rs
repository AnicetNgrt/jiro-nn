use crate::{
    datatable::DataTable,
    linalg::{Matrix, Scalar},
    vision::image::Image,
};

use self::model::{impl_model_from_model_fields, Model};

pub mod batched_columns_activation;
pub mod batched_columns_dense_layer;
pub mod batched_columns_tanh;
pub mod loss;
pub mod mapping;
pub mod matrix_learnable_adam;
pub mod matrix_learnable_momentum;
pub mod matrix_learnable_sgd;
pub mod model;
pub mod optimizer;
pub mod combinatory_op;
pub mod model_op_builder;

pub trait Data: 'static {}

pub trait LearnableOp<D: Data>: Model {
    fn forward_inference(&mut self, input: D) -> D;
    fn forward(&mut self, input: D) -> D;
    fn backward(&mut self, incoming_grad: D) -> D;
}

pub trait InputTransformationOp<DataIn: Data, DataOut: Data>: Model {
    fn transform(&mut self, input: DataIn) -> DataOut;
    fn revert(&mut self, output: DataOut) -> DataIn;
}

pub trait ReferenceTransformationOp<DataIn: Data, DataOut: Data>: Model {
    fn transform(&mut self, reference: DataIn) -> DataOut;
    fn revert(&mut self, reference: DataOut) -> DataIn;
}

pub trait TotalTransformationOp<DataIn: Data, DataOut: Data>: Model {
    fn transform(&mut self, input_or_reference: DataIn) -> DataOut;
    fn revert(&mut self, output_or_reference: DataOut) -> DataIn;
}

pub trait ModelOp<DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>: Model {
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

macro_rules! impl_model_op_for_learnable_op {
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

pub(crate) use impl_model_op_for_learnable_op;

macro_rules! impl_model_op_for_input_transformation_op {
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

pub(crate) use impl_model_op_for_input_transformation_op;

macro_rules! impl_model_op_for_reference_transformation_op {
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

pub(crate) use impl_model_op_for_reference_transformation_op;

macro_rules! impl_model_op_for_total_transformation_op {
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

pub(crate) use impl_model_op_for_total_transformation_op;

pub struct OpChain<
    'a,
    DataIn: Data,
    DataMid: Data,
    DataOut: Data,
    DataRefIn: Data,
    DataRefMid: Data,
    DataRefOut: Data,
> {
    first_op: Box<dyn ModelOp<DataIn, DataMid, DataRefIn, DataRefMid> + 'a>,
    second_op: Box<dyn ModelOp<DataMid, DataOut, DataRefMid, DataRefOut> + 'a>,
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > Model for OpChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    impl_model_from_model_fields!(first_op, second_op);
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>
    for OpChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
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
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > OpChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn ModelOp<DataIn, DataMid, DataRefIn, DataRefMid> + 'a>,
        second_op: Box<dyn ModelOp<DataMid, DataOut, DataRefMid, DataRefOut> + 'a>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<D: Data> Data for Vec<D> {}
impl Data for bool {}
impl Data for Scalar {}
impl Data for Matrix {}
impl Data for Image {}
impl Data for DataTable {}
