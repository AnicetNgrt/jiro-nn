use crate::linalg::Scalar;

use crate::ops::{
    model::{impl_model_no_params, Model},
    op_graphs::op_node::{
        impl_op_node_for_input_transformation_op, InputTransformationOp, OpNodeTrait,
    },
    Data,
};

pub struct InputMappingOp<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    f: F,
    fp: FP,
    fm: FM,
    _phantom: std::marker::PhantomData<&'g (DataIn, DataOut)>,
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>
    InputMappingOp<'g, DataIn, DataOut, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    pub fn new(mapper: F, mapper_reversed: FP, meta_mapper: FM) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
            fm: meta_mapper,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn map_meta(&self, meta: DataIn::Meta) -> DataOut::Meta {
        (self.fm)(meta)
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM> Model
    for InputMappingOp<'g, DataIn, DataOut, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    impl_model_no_params!();
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM> InputTransformationOp<'g, DataIn, DataOut>
    for InputMappingOp<'g, DataIn, DataOut, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    fn transform(&mut self, input: DataIn) -> DataOut {
        (self.f)(input)
    }

    fn revert(&mut self, output: DataOut) -> DataIn {
        (self.fp)(output)
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRef: Data<'g>, F, FP, FM>
    OpNodeTrait<'g, DataIn, DataOut, DataRef, DataRef>
    for InputMappingOp<'g, DataIn, DataOut, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    impl_op_node_for_input_transformation_op!(DataIn, DataOut, DataRef, DataRef);
}
