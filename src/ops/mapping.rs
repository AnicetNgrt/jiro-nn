use crate::linalg::Scalar;

use super::{
    model::{impl_model_no_params, Model},
    Data, InputTransformationOp, ReferenceTransformationOp, TotalTransformationOp, op_graph::{OpSubgraphTrait, impl_op_subgraph_for_input_transformation_op, impl_op_subgraph_for_reference_transformation_op, impl_op_subgraph_for_total_transformation_op},
};

pub struct InputMappingOp<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<&'g (DataIn, DataOut)>,
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
    InputMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    pub fn new(mapper: F, mapper_reversed: FP) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP> Model
    for InputMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_no_params!();
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
    InputTransformationOp<'g, DataIn, DataOut>
    for InputMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    fn transform(&mut self, input: DataIn) -> DataOut {
        (self.f)(input)
    }

    fn revert(&mut self, output: DataOut) -> DataIn {
        (self.fp)(output)
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRef: Data<'g>, F, FP>
    OpSubgraphTrait<'g, DataIn, DataOut, DataRef, DataRef>
    for InputMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_op_subgraph_for_input_transformation_op!(DataIn, DataOut, DataRef, DataRef);
}

macro_rules! impl_op_builder_from_input_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<'g, DataRef: Data<'g>>
            OpSubgraphBuilder<'g, $in_type, $out_type, DataRef, DataRef> for $t
        {
            fn build(
                &mut self,
                sample_data: $in_type,
                sample_ref: DataRef,
            ) -> (
                Box<dyn OpSubgraphTrait<'g, $in_type, $out_type, DataRef, DataRef> + 'g>,
                ($in_type, DataRef),
            ) {
                let op = InputMappingOp::new($transform, $revert);

                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_input_transformation_closures;

pub struct ReferenceMappingOp<
    'g,
    DataRefIn: Data<'g>,
    DataRefOut: Data<'g>,
    F,
    FP,
> where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<&'g (DataRefIn, DataRefOut)>,
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP>
    ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    pub fn new(mapper: F, mapper_reversed: FP) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP> Model
    for ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    impl_model_no_params!();
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP>
    ReferenceTransformationOp<'g, DataRefIn, DataRefOut>
    for ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    fn transform(&mut self, reference: DataRefIn) -> DataRefOut {
        (self.f)(reference)
    }

    fn revert(&mut self, reference: DataRefOut) -> DataRefIn {
        (self.fp)(reference)
    }
}

impl<'g, D: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>, F, FP>
    OpSubgraphTrait<'g, D, D, DataRefIn, DataRefOut>
    for ReferenceMappingOp<'g, DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    impl_op_subgraph_for_reference_transformation_op!(D, D, DataRefIn, DataRefOut);
}

macro_rules! impl_op_builder_from_reference_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<'g, D: Data<'g>> OpSubgraphBuilder<'g, D, D, $in_type, $out_type>
            for $t
        {
            fn build(
                &mut self,
                sample_data: D,
                sample_ref: $in_type,
            ) -> (
                Box<dyn OpSubgraphTrait<'g, D, D, $in_type, $out_type> + 'g>,
                (D, $in_type),
            ) {
                let op = ReferenceMappingOp::new($transform, $revert);

                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_reference_transformation_closures;

pub struct TotalMappingOp<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<&'g (DataIn, DataOut)>,
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
    TotalMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    pub fn new(mapper: F, mapper_reversed: FP) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP> Model
    for TotalMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_no_params!();
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
    TotalTransformationOp<'g, DataIn, DataOut>
    for TotalMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    fn transform(&mut self, input: DataIn) -> DataOut {
        (self.f)(input)
    }

    fn revert(&mut self, output: DataOut) -> DataIn {
        (self.fp)(output)
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP>
    OpSubgraphTrait<'g, DataIn, DataOut, DataIn, DataOut>
    for TotalMappingOp<'g, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_op_subgraph_for_total_transformation_op!(DataIn, DataOut, DataIn, DataOut);
}

macro_rules! impl_op_builder_from_total_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<'g> OpSubgraphBuilder<'g, $in_type, $out_type, $in_type, $out_type>
            for $t
        {
            fn build(
                &mut self,
                sample_data: $in_type,
                sample_ref: $in_type,
            ) -> (
                Box<dyn OpSubgraphTrait<'g, $in_type, $out_type, $in_type, $out_type> + 'g>,
                ($in_type, $in_type),
            ) {
                let op = TotalMappingOp::new($transform, $revert);

                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_total_transformation_closures;
