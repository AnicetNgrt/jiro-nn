use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    datatable::DataTable,
    linalg::{Matrix, Scalar},
    vision::image::Image,
};

use self::{
    model::{impl_model_from_model_fields, Model},
    op_graph::OpSubgraphTrait
};

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
pub mod op_graph_builder;
pub mod optimizer;
pub mod vec_to_matrix;
pub mod op_graph;

pub trait Data<'g>: 'g {}

pub trait LearnableOp<'g, D: Data<'g>>: Model {
    fn forward_inference(&mut self, input: D) -> D;
    fn forward(&mut self, input: D) -> D;
    fn backward(&mut self, incoming_grad: D) -> D;
}

pub trait InputTransformationOp<'g, DataIn: Data<'g>, DataOut: Data<'g>>:
    Model
{
    fn transform(&mut self, input: DataIn) -> DataOut;
    fn revert(&mut self, output: DataOut) -> DataIn;
}

pub trait ReferenceTransformationOp<'g, DataIn: Data<'g>, DataOut: Data<'g>>:
    Model
{
    fn transform(&mut self, reference: DataIn) -> DataOut;
    fn revert(&mut self, reference: DataOut) -> DataIn;
}

pub trait TotalTransformationOp<'g, DataIn: Data<'g>, DataOut: Data<'g>>:
    Model
{
    fn transform(&mut self, input_or_reference: DataIn) -> DataOut;
    fn revert(&mut self, output_or_reference: DataOut) -> DataIn;
}

pub struct OpChain<
    'g,
    DataIn: Data<'g>,
    DataMid: Data<'g>,
    DataOut: Data<'g>,
    DataRefIn: Data<'g>,
    DataRefMid: Data<'g>,
    DataRefOut: Data<'g>,
> {
    first_op: Box<dyn OpSubgraphTrait<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
    second_op: Box<dyn OpSubgraphTrait<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
}

impl<
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > Model for OpChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
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
    > OpSubgraphTrait<'g, DataIn, DataOut, DataRefIn, DataRefOut>
    for OpChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
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
        'g,
        DataIn: Data<'g>,
        DataMid: Data<'g>,
        DataOut: Data<'g>,
        DataRefIn: Data<'g>,
        DataRefMid: Data<'g>,
        DataRefOut: Data<'g>,
    > OpChain<'g, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn OpSubgraphTrait<'g, DataIn, DataMid, DataRefIn, DataRefMid> + 'g>,
        second_op: Box<dyn OpSubgraphTrait<'g, DataMid, DataOut, DataRefMid, DataRefOut> + 'g>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<'g, D: Data<'g>> Data<'g> for &'g D {}
impl<'g, D: Data<'g>> Data<'g> for RefCell<D> {}
impl<'g, D: Data<'g>> Data<'g> for Arc<D> {}
impl<'g, D: Data<'g>> Data<'g> for Rc<D> {}
impl<'g, D: Data<'g>> Data<'g> for Box<D> {}
impl<'g, D: Data<'g>> Data<'g> for Option<D> {}
impl<
        'g,
        D1: Data<'g>,
        D2: Data<'g>,
        D3: Data<'g>,
        D4: Data<'g>,
        D5: Data<'g>,
        D6: Data<'g>,
    > Data<'g> for (D1, D2, D3, D4, D5, D6)
{
}
impl<
        'g,
        D1: Data<'g>,
        D2: Data<'g>,
        D3: Data<'g>,
        D4: Data<'g>,
        D5: Data<'g>,
    > Data<'g> for (D1, D2, D3, D4, D5)
{
}
impl<'g, D1: Data<'g>, D2: Data<'g>, D3: Data<'g>, D4: Data<'g>>
    Data<'g> for (D1, D2, D3, D4)
{
}
impl<'g, D1: Data<'g>, D2: Data<'g>, D3: Data<'g>> Data<'g>
    for (D1, D2, D3)
{
}
impl<'g, D1: Data<'g>, D2: Data<'g>> Data<'g> for (D1, D2) {}
impl<'g, D: Data<'g>> Data<'g> for (D,) {}
impl<'g> Data<'g> for () {}
impl<'g, D: Data<'g>> Data<'g> for Vec<D> {}
impl<'g, D: Data<'g>, const S: usize> Data<'g> for [D; S] {}
impl<'g> Data<'g> for bool {}
impl<'g> Data<'g> for String {}
impl<'g> Data<'g> for usize {}
impl<'g> Data<'g> for Scalar {}
impl<'g> Data<'g> for Matrix {}
impl<'g> Data<'g> for Image {}
impl<'g> Data<'g> for DataTable {}
