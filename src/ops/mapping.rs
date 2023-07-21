use crate::linalg::Scalar;

use super::{
    impl_model_op_for_input_transformation_op, impl_model_op_for_reference_transformation_op,
    impl_model_op_for_total_transformation_op,
    model::{impl_model_no_params, Model},
    Data, InputTransformationOp, ModelOp, ReferenceTransformationOp, TotalTransformationOp,
};

pub struct InputMappingOp<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<&'opgraph (DataIn, DataOut)>,
}

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
    InputMappingOp<'opgraph, DataIn, DataOut, F, FP>
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

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP> Model
    for InputMappingOp<'opgraph, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_no_params!();
}

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
    InputTransformationOp<'opgraph, DataIn, DataOut>
    for InputMappingOp<'opgraph, DataIn, DataOut, F, FP>
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

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, DataRef: Data<'opgraph>, F, FP>
    ModelOp<'opgraph, DataIn, DataOut, DataRef, DataRef>
    for InputMappingOp<'opgraph, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_op_for_input_transformation_op!(DataIn, DataOut, DataRef, DataRef);
}

macro_rules! impl_op_builder_from_input_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<'opgraph, DataRef: Data<'opgraph>>
            OpSubgraphBuilder<'opgraph, $in_type, $out_type, DataRef, DataRef> for $t
        {
            fn build(
                &mut self,
                sample_data: $in_type,
                sample_ref: DataRef,
            ) -> (
                Box<dyn ModelOp<'opgraph, $in_type, $out_type, DataRef, DataRef> + 'opgraph>,
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
    'opgraph,
    DataRefIn: Data<'opgraph>,
    DataRefOut: Data<'opgraph>,
    F,
    FP,
> where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<&'opgraph (DataRefIn, DataRefOut)>,
}

impl<'opgraph, DataRefIn: Data<'opgraph>, DataRefOut: Data<'opgraph>, F, FP>
    ReferenceMappingOp<'opgraph, DataRefIn, DataRefOut, F, FP>
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

impl<'opgraph, DataRefIn: Data<'opgraph>, DataRefOut: Data<'opgraph>, F, FP> Model
    for ReferenceMappingOp<'opgraph, DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    impl_model_no_params!();
}

impl<'opgraph, DataRefIn: Data<'opgraph>, DataRefOut: Data<'opgraph>, F, FP>
    ReferenceTransformationOp<'opgraph, DataRefIn, DataRefOut>
    for ReferenceMappingOp<'opgraph, DataRefIn, DataRefOut, F, FP>
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

impl<'opgraph, D: Data<'opgraph>, DataRefIn: Data<'opgraph>, DataRefOut: Data<'opgraph>, F, FP>
    ModelOp<'opgraph, D, D, DataRefIn, DataRefOut>
    for ReferenceMappingOp<'opgraph, DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    impl_model_op_for_reference_transformation_op!(D, D, DataRefIn, DataRefOut);
}

macro_rules! impl_op_builder_from_reference_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<'opgraph, D: Data<'opgraph>> OpSubgraphBuilder<'opgraph, D, D, $in_type, $out_type>
            for $t
        {
            fn build(
                &mut self,
                sample_data: D,
                sample_ref: $in_type,
            ) -> (
                Box<dyn ModelOp<'opgraph, D, D, $in_type, $out_type> + 'opgraph>,
                (D, $in_type),
            ) {
                let op = ReferenceMappingOp::new($transform, $revert);

                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_reference_transformation_closures;

pub struct TotalMappingOp<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<&'opgraph (DataIn, DataOut)>,
}

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
    TotalMappingOp<'opgraph, DataIn, DataOut, F, FP>
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

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP> Model
    for TotalMappingOp<'opgraph, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_no_params!();
}

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
    TotalTransformationOp<'opgraph, DataIn, DataOut>
    for TotalMappingOp<'opgraph, DataIn, DataOut, F, FP>
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

impl<'opgraph, DataIn: Data<'opgraph>, DataOut: Data<'opgraph>, F, FP>
    ModelOp<'opgraph, DataIn, DataOut, DataIn, DataOut>
    for TotalMappingOp<'opgraph, DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_op_for_total_transformation_op!(DataIn, DataOut, DataIn, DataOut);
}

macro_rules! impl_op_builder_from_total_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<'opgraph> OpSubgraphBuilder<'opgraph, $in_type, $out_type, $in_type, $out_type>
            for $t
        {
            fn build(
                &mut self,
                sample_data: $in_type,
                sample_ref: $in_type,
            ) -> (
                Box<dyn ModelOp<'opgraph, $in_type, $out_type, $in_type, $out_type> + 'opgraph>,
                ($in_type, $in_type),
            ) {
                let op = TotalMappingOp::new($transform, $revert);

                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_total_transformation_closures;
