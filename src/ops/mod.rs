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

pub mod batched_columns_activation;
pub mod batched_columns_dense_layer;
pub mod batched_columns_tanh;
pub mod combinatory_op;
pub mod learning_rate;
pub mod loss;
pub mod mappings;
pub mod matrix_learnable_adam;
pub mod matrix_learnable_momentum;
pub mod matrix_learnable_sgd;
pub mod model;
pub mod op_graph_builder;
pub mod op_graphs;
pub mod optimizer;
pub mod transformations;
pub mod vec_to_matrix;

pub trait Data<'g>: 'g {
    type Meta;
    fn meta(&self) -> Self::Meta;
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
