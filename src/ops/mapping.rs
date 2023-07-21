use crate::linalg::Scalar;

use super::{
    impl_model_op_for_input_transformation_op, impl_model_op_for_reference_transformation_op,
    impl_model_op_for_total_transformation_op,
    model::{impl_model_no_params, Model},
    Data, InputTransformationOp, ModelOp, ReferenceTransformationOp, TotalTransformationOp,
};

pub struct InputMappingOp<DataIn: Data, DataOut: Data, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<(DataIn, DataOut)>,
}

impl<DataIn: Data, DataOut: Data, F, FP> InputMappingOp<DataIn, DataOut, F, FP>
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

impl<DataIn: Data, DataOut: Data, F, FP> Model for InputMappingOp<DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_no_params!();
}

impl<DataIn: Data, DataOut: Data, F, FP> InputTransformationOp<DataIn, DataOut>
    for InputMappingOp<DataIn, DataOut, F, FP>
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

impl<DataIn: Data, DataOut: Data, DataRef: Data, F, FP> ModelOp<DataIn, DataOut, DataRef, DataRef>
    for InputMappingOp<DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_op_for_input_transformation_op!(DataIn, DataOut, DataRef, DataRef);
}

macro_rules! impl_op_builder_from_input_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<DataRef: Data> OpBuilder<$in_type, $out_type, DataRef, DataRef> for $t {
            fn build(
                &mut self,
                sample_data: $in_type,
                sample_ref: DataRef,
            ) -> (
                Box<dyn ModelOp<$in_type, $out_type, DataRef, DataRef>>,
                ($in_type, DataRef),
            ) {
                let op = InputMappingOp::new(
                    $transform,
                    $revert,
                );
        
                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_input_transformation_closures;

pub struct ReferenceMappingOp<DataRefIn: Data, DataRefOut: Data, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<(DataRefIn, DataRefOut)>,
}

impl<DataRefIn: Data, DataRefOut: Data, F, FP> ReferenceMappingOp<DataRefIn, DataRefOut, F, FP>
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

impl<DataRefIn: Data, DataRefOut: Data, F, FP> Model
    for ReferenceMappingOp<DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    impl_model_no_params!();
}

impl<DataRefIn: Data, DataRefOut: Data, F, FP> ReferenceTransformationOp<DataRefIn, DataRefOut>
    for ReferenceMappingOp<DataRefIn, DataRefOut, F, FP>
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

impl<D: Data, DataRefIn: Data, DataRefOut: Data, F, FP> ModelOp<D, D, DataRefIn, DataRefOut>
    for ReferenceMappingOp<DataRefIn, DataRefOut, F, FP>
where
    F: Fn(DataRefIn) -> DataRefOut,
    FP: Fn(DataRefOut) -> DataRefIn,
{
    impl_model_op_for_reference_transformation_op!(D, D, DataRefIn, DataRefOut);
}

macro_rules! impl_op_builder_from_reference_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl<D: Data> OpBuilder<D, D, $in_type, $out_type> for $t {
            fn build(
                &mut self,
                sample_data: D,
                sample_ref: $in_type,
            ) -> (
                Box<dyn ModelOp<D, D, $in_type, $out_type>>,
                (D, $in_type),
            ) {
                let op = ReferenceMappingOp::new(
                    $transform,
                    $revert,
                );
        
                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_reference_transformation_closures;

pub struct TotalMappingOp<DataIn: Data, DataOut: Data, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    f: F,
    fp: FP,
    _phantom: std::marker::PhantomData<(DataIn, DataOut)>,
}

impl<DataIn: Data, DataOut: Data, F, FP> TotalMappingOp<DataIn, DataOut, F, FP>
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

impl<DataIn: Data, DataOut: Data, F, FP> Model for TotalMappingOp<DataIn, DataOut, F, FP>
where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_no_params!();
}

impl<DataIn: Data, DataOut: Data, F, FP> TotalTransformationOp<DataIn, DataOut>
    for TotalMappingOp<DataIn, DataOut, F, FP>
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

impl<DataIn: Data, DataOut: Data, F, FP> ModelOp<DataIn, DataOut, DataIn, DataOut>
    for TotalMappingOp<DataIn, DataOut, F, FP>
    where
    F: Fn(DataIn) -> DataOut,
    FP: Fn(DataOut) -> DataIn,
{
    impl_model_op_for_total_transformation_op!(DataIn, DataOut, DataIn, DataOut);
}

macro_rules! impl_op_builder_from_total_transformation_closures {
    ($t:ty, $in_type:ty, $out_type:ty, $transform:tt, $revert:tt) => {
        impl OpBuilder<$in_type, $out_type, $in_type, $out_type> for $t {
            fn build(
                &mut self,
                sample_data: $in_type,
                sample_ref: $in_type,
            ) -> (
                Box<dyn ModelOp<$in_type, $out_type, $in_type, $out_type>>,
                ($in_type, $in_type),
            ) {
                let op = TotalMappingOp::new(
                    $transform,
                    $revert,
                );
        
                (Box::new(op), (sample_data, sample_ref))
            }
        }
    };
}

pub(crate) use impl_op_builder_from_total_transformation_closures;