use crate::linalg::Scalar;

use crate::ops::op_graph::impl_op_subgraph_for_total_transformation_op;
use crate::ops::{
    model::{impl_model_no_params, Model},
    op_graph::OpSubgraphTrait,
    Data, TotalTransformationOp,
};

pub struct TotalMappingOp<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>
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
    TotalMappingOp<'g, DataIn, DataOut, F, FP, FM>
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
    for TotalMappingOp<'g, DataIn, DataOut, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    impl_model_no_params!();
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM> TotalTransformationOp<'g, DataIn, DataOut>
    for TotalMappingOp<'g, DataIn, DataOut, F, FP, FM>
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

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>
    OpSubgraphTrait<'g, DataIn, DataOut, DataIn, DataOut>
    for TotalMappingOp<'g, DataIn, DataOut, F, FP, FM>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
    FM: Fn(DataIn::Meta) -> DataOut::Meta,
{
    impl_op_subgraph_for_total_transformation_op!(DataIn, DataOut, DataIn, DataOut);
}

macro_rules! impl_op_builder_from_total_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt, $meta:tt) => {
        impl<'g> OpSubgraphBuilder<'g, $in_type, $out_type, $in_type, $out_type>
            for $t
        {
            fn build(
                &mut self,
                meta_data: <$in_type as Data>::Meta,
                meta_ref: <$in_type as Data>::Meta,
            ) -> (
                Box<dyn OpSubgraphTrait<'g, $in_type, $out_type, $in_type, $out_type> + 'g>,
                (<$out_type as Data>::Meta, <$out_type as Data>::Meta),
            ) {
                let op = TotalMappingOp::new($transform, $revert, $meta);
                let meta_data = op.map_meta(meta_data);
                let meta_ref = op.map_meta(meta_ref);

                (Box::new(op), (meta_data, meta_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_total_transformation_closures;
