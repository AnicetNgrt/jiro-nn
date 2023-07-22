use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::{
    datatable::DataTable,
    linalg::{Matrix, MatrixTrait, Scalar},
    vision::image::{Image, ImageTrait},
};

use self::{
    model::{impl_model_from_model_fields, Model},
    op_graph::OpSubgraphTrait,
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
pub mod op_graph;
pub mod op_graph_builder;
pub mod optimizer;
pub mod vec_to_matrix;

pub trait Data<'g>: 'g {
    type Meta;
    fn meta(&self) -> Self::Meta;
}

pub trait LearnableOp<'g, D: Data<'g>>: Model {
    fn forward_inference(&mut self, input: D) -> D;
    fn forward(&mut self, input: D) -> D;
    fn backward(&mut self, incoming_grad: D) -> D;
}

pub trait InputTransformationOp<'g, DataIn: Data<'g>, DataOut: Data<'g>>: Model {
    fn transform(&mut self, input: DataIn) -> DataOut;
    fn revert(&mut self, output: DataOut) -> DataIn;
}

pub trait ReferenceTransformationOp<'g, DataIn: Data<'g>, DataOut: Data<'g>>: Model {
    fn transform(&mut self, reference: DataIn) -> DataOut;
    fn revert(&mut self, reference: DataOut) -> DataIn;
}

pub trait TotalTransformationOp<'g, DataIn: Data<'g>, DataOut: Data<'g>>: Model {
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

impl<'g, D: Data<'g>> Data<'g> for &'g D {
    type Meta = D::Meta;

    fn meta(&self) -> Self::Meta {
        (*self).meta()
    }
}

impl<'g, D: Data<'g>> Data<'g> for Mutex<D> {
    type Meta = D::Meta;

    fn meta(&self) -> Self::Meta {
        self.lock().unwrap().meta()
    }
}

impl<'g, D: Data<'g>> Data<'g> for RefCell<D> {
    type Meta = D::Meta;

    fn meta(&self) -> Self::Meta {
        self.borrow().meta()
    }
}
impl<'g, D: Data<'g>> Data<'g> for Arc<D> {
    type Meta = D::Meta;

    fn meta(&self) -> Self::Meta {
        self.as_ref().meta()
    }
}
impl<'g, D: Data<'g>> Data<'g> for Rc<D> {
    type Meta = D::Meta;

    fn meta(&self) -> Self::Meta {
        self.as_ref().meta()
    }
}
impl<'g, D: Data<'g>> Data<'g> for Box<D> {
    type Meta = D::Meta;

    fn meta(&self) -> Self::Meta {
        self.as_ref().meta()
    }
}
impl<'g, D: Data<'g>> Data<'g> for Option<D> {
    type Meta = Option<D::Meta>;

    fn meta(&self) -> Self::Meta {
        self.as_ref().map(|d| d.meta())
    }
}
impl<'g, D1: Data<'g>, D2: Data<'g>, D3: Data<'g>, D4: Data<'g>, D5: Data<'g>, D6: Data<'g>>
    Data<'g> for (D1, D2, D3, D4, D5, D6)
{
    type Meta = (D1::Meta, D2::Meta, D3::Meta, D4::Meta, D5::Meta, D6::Meta);

    fn meta(&self) -> Self::Meta {
        (
            self.0.meta(),
            self.1.meta(),
            self.2.meta(),
            self.3.meta(),
            self.4.meta(),
            self.5.meta(),
        )
    }
}
impl<'g, D1: Data<'g>, D2: Data<'g>, D3: Data<'g>, D4: Data<'g>, D5: Data<'g>> Data<'g>
    for (D1, D2, D3, D4, D5)
{
    type Meta = (D1::Meta, D2::Meta, D3::Meta, D4::Meta, D5::Meta);

    fn meta(&self) -> Self::Meta {
        (
            self.0.meta(),
            self.1.meta(),
            self.2.meta(),
            self.3.meta(),
            self.4.meta(),
        )
    }
}
impl<'g, D1: Data<'g>, D2: Data<'g>, D3: Data<'g>, D4: Data<'g>> Data<'g> for (D1, D2, D3, D4) {
    type Meta = (D1::Meta, D2::Meta, D3::Meta, D4::Meta);

    fn meta(&self) -> Self::Meta {
        (self.0.meta(), self.1.meta(), self.2.meta(), self.3.meta())
    }
}
impl<'g, D1: Data<'g>, D2: Data<'g>, D3: Data<'g>> Data<'g> for (D1, D2, D3) {
    type Meta = (D1::Meta, D2::Meta, D3::Meta);

    fn meta(&self) -> Self::Meta {
        (self.0.meta(), self.1.meta(), self.2.meta())
    }
}
impl<'g, D1: Data<'g>, D2: Data<'g>> Data<'g> for (D1, D2) {
    type Meta = (D1::Meta, D2::Meta);

    fn meta(&self) -> Self::Meta {
        (self.0.meta(), self.1.meta())
    }
}
impl<'g, D: Data<'g>> Data<'g> for (D,) {
    type Meta = (D::Meta,);

    fn meta(&self) -> Self::Meta {
        (self.0.meta(),)
    }
}
impl<'g> Data<'g> for () {
    type Meta = ();

    fn meta(&self) -> Self::Meta {
        ()
    }
}
impl<'g, D: Data<'g>> Data<'g> for Vec<D> {
    type Meta = Vec<D::Meta>;

    fn meta(&self) -> Self::Meta {
        self.iter().map(|d| d.meta()).collect()
    }
}

impl<'g, D: Data<'g>, const S: usize> Data<'g> for [D; S]
where
    // could get rid of this with arrayvec or nightly's feature(array_map)
    D::Meta: Copy,
{
    type Meta = [D::Meta; S];

    fn meta(&self) -> Self::Meta {
        let mut res = [self[0].meta(); S];
        for i in 0..S {
            res[i] = self[i].meta();
        }
        res
    }
}
impl<'g> Data<'g> for bool {
    type Meta = ();

    fn meta(&self) -> Self::Meta {
        ()
    }
}
impl<'g> Data<'g> for String {
    type Meta = usize;

    fn meta(&self) -> Self::Meta {
        self.len()
    }
}
impl<'g> Data<'g> for usize {
    type Meta = ();

    fn meta(&self) -> Self::Meta {
        ()
    }
}
impl<'g> Data<'g> for Scalar {
    type Meta = ();

    fn meta(&self) -> Self::Meta {
        ()
    }
}
impl<'g> Data<'g> for Matrix {
    type Meta = (usize, usize);

    fn meta(&self) -> Self::Meta {
        self.dim()
    }
}
impl<'g> Data<'g> for Image {
    type Meta = (usize, usize, usize, usize);

    fn meta(&self) -> Self::Meta {
        let (nrows, ncols, nchans) = self.image_dims();
        let nsamples = self.samples();
        (nrows, ncols, nchans, nsamples)
    }
}
impl<'g> Data<'g> for DataTable {
    type Meta = ();
    fn meta(&self) -> Self::Meta {
        ()
    }
}
