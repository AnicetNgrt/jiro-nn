use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    datatable::DataTable,
    linalg::{Matrix, Scalar},
    vision::image::Image,
};

use self::model::{impl_model_from_model_fields, Model};

pub mod batched_columns_activation;
pub mod batched_columns_dense_layer;
pub mod batched_columns_tanh;
pub mod combinatory_op;
pub mod learning_rate;
pub mod loss;
pub mod mapping;
pub mod matrix_learnable_adam;
pub mod matrix_learnable_momentum;
pub mod matrix_learnable_sgd;
pub mod model;
pub mod model_op_builder;
pub mod optimizer;
pub mod vec_to_matrix;

pub trait Data<'opgraph>: 'opgraph {}

pub trait LearnableOp<'opgraph, D: Data<'opgraph>>: Model {
    fn forward_inference(&mut self, input: D) -> D;
    fn forward(&mut self, input: D) -> D;
    fn backward(&mut self, incoming_grad: D) -> D;
}

pub trait InputTransformationOp<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>>:
    Model
{
    fn transform(&mut self, input: DataIn) -> DataOut;
    fn revert(&mut self, output: DataOut) -> DataIn;
}

pub trait ReferenceTransformationOp<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>>:
    Model
{
    fn transform(&mut self, reference: DataIn) -> DataOut;
    fn revert(&mut self, reference: DataOut) -> DataIn;
}

pub trait TotalTransformationOp<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>>:
    Model
{
    fn transform(&mut self, input_or_reference: DataIn) -> DataOut;
    fn revert(&mut self, output_or_reference: DataOut) -> DataIn;
}

pub trait ModelOp<
    'opgraph,
    DataIn: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
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
    'opgraph,
    DataIn: Data<'opgraph>,
    DataMid: Data<'opgraph>,
    DataOut: Data<'opgraph>,
    DataRefIn: Data<'opgraph>,
    DataRefMid: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
> {
    first_op: Box<dyn ModelOp<'opgraph, DataIn, DataMid, DataRefIn, DataRefMid> + 'opgraph>,
    second_op: Box<dyn ModelOp<'opgraph, DataMid, DataOut, DataRefMid, DataRefOut> + 'opgraph>,
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataMid: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefMid: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > Model for OpChain<'opgraph, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    impl_model_from_model_fields!(first_op, second_op);
}

impl<
        'opgraph,
        DataIn: Data<'opgraph>,
        DataMid: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefMid: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > ModelOp<'opgraph, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpChain<'opgraph, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
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
        'opgraph,
        DataIn: Data<'opgraph>,
        DataMid: Data<'opgraph>,
        DataOut: Data<'opgraph>,
        DataRefIn: Data<'opgraph>,
        DataRefMid: Data<'opgraph>,
        DataRefOut: Data<'opgraph>,
    > OpChain<'opgraph, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn ModelOp<'opgraph, DataIn, DataMid, DataRefIn, DataRefMid> + 'opgraph>,
        second_op: Box<dyn ModelOp<'opgraph, DataMid, DataOut, DataRefMid, DataRefOut> + 'opgraph>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for &'opgraph D {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for RefCell<D> {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for Arc<D> {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for Rc<D> {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for Box<D> {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for Option<D> {}
impl<
        'opgraph,
        D1: Data<'opgraph>,
        D2: Data<'opgraph>,
        D3: Data<'opgraph>,
        D4: Data<'opgraph>,
        D5: Data<'opgraph>,
        D6: Data<'opgraph>,
    > Data<'opgraph> for (D1, D2, D3, D4, D5, D6)
{
}
impl<
        'opgraph,
        D1: Data<'opgraph>,
        D2: Data<'opgraph>,
        D3: Data<'opgraph>,
        D4: Data<'opgraph>,
        D5: Data<'opgraph>,
    > Data<'opgraph> for (D1, D2, D3, D4, D5)
{
}
impl<'opgraph, D1: Data<'opgraph>, D2: Data<'opgraph>, D3: Data<'opgraph>, D4: Data<'opgraph>>
    Data<'opgraph> for (D1, D2, D3, D4)
{
}
impl<'opgraph, D1: Data<'opgraph>, D2: Data<'opgraph>, D3: Data<'opgraph>> Data<'opgraph>
    for (D1, D2, D3)
{
}
impl<'opgraph, D1: Data<'opgraph>, D2: Data<'opgraph>> Data<'opgraph> for (D1, D2) {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for (D,) {}
impl<'opgraph> Data<'opgraph> for () {}
impl<'opgraph, D: Data<'opgraph>> Data<'opgraph> for Vec<D> {}
impl<'opgraph, D: Data<'opgraph>, const S: usize> Data<'opgraph> for [D; S] {}
impl<'opgraph> Data<'opgraph> for bool {}
impl<'opgraph> Data<'opgraph> for String {}
impl<'opgraph> Data<'opgraph> for usize {}
impl<'opgraph> Data<'opgraph> for Scalar {}
impl<'opgraph> Data<'opgraph> for Matrix {}
impl<'opgraph> Data<'opgraph> for Image {}
impl<'opgraph> Data<'opgraph> for DataTable {}
