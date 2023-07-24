use crate::linalg::Scalar;

use crate::ops::{
    model::{impl_model_no_params, Model},
    op_graphs::op_node::{
        impl_op_node_for_reference_transformation_op, OpNodeTrait, ReferenceTransformationOp,
    },
    Data,
};

pub struct ReferenceMappingOp<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP, FM>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
    FM: Fn(DataRefIn::Meta) -> DataRefOut::Meta,
{
    f: F,
    fp: FP,
    fm: FM,
    _phantom: std::marker::PhantomData<&'g (DataRefIn, DataRefOut)>,
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP, FM>
    ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP, FM>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
    FM: Fn(DataRefIn::Meta) -> DataRefOut::Meta,
{
    pub fn new(mapper: F, mapper_reversed: FP, meta_mapper: FM) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
            fm: meta_mapper,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn map_meta(&self, meta: DataRefIn::Meta) -> DataRefOut::Meta {
        (self.fm)(meta)
    }
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP, FM> Model
    for ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP, FM>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
    FM: Fn(DataRefIn::Meta) -> DataRefOut::Meta,
{
    impl_model_no_params!();
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP, FM>
    ReferenceTransformationOp<'g, DataRefIn, DataRefOut>
    for ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP, FM>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
    FM: Fn(DataRefIn::Meta) -> DataRefOut::Meta,
{
    fn transform(&mut self, reference: DataRefIn) -> DataRefOut {
        (self.f)(reference)
    }

    fn revert(&mut self, output: DataRefOut) -> DataRefIn {
        (self.fp)(output)
    }
}

impl<'g, D: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP, FM>
    OpNodeTrait<'g, D, D, DataRefIn, DataRefOut>
    for ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP, FM>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
    FM: Fn(DataRefIn::Meta) -> DataRefOut::Meta,
{
    impl_op_node_for_reference_transformation_op!(D, D, DataRefIn, DataRefOut);
}
