use crate::linalg::Scalar;

use super::{
    impl_model_op_for_input_transformation_op, impl_model_op_for_reference_transformation_op,
    impl_model_op_for_total_transformation_op,
    model::{impl_model_no_params, Model},
    Data, InputTransformationOp, ModelOp, ReferenceTransformationOp, TotalTransformationOp,
};

pub struct InputMappingOp<DataIn: Data, DataOut: Data> {
    f: fn(DataIn) -> DataOut,
    fp: fn(DataOut) -> DataIn,
}

impl<DataIn: Data, DataOut: Data> InputMappingOp<DataIn, DataOut> {
    pub fn new(mapper: fn(DataIn) -> DataOut, mapper_reversed: fn(DataOut) -> DataIn) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
        }
    }
}

impl<DataIn: Data, DataOut: Data> Model for InputMappingOp<DataIn, DataOut> {
    impl_model_no_params!();
}

impl<DataIn: Data, DataOut: Data> InputTransformationOp<DataIn, DataOut>
    for InputMappingOp<DataIn, DataOut>
{
    fn transform(&mut self, input: DataIn) -> DataOut {
        (self.f)(input)
    }

    fn revert(&mut self, output: DataOut) -> DataIn {
        (self.fp)(output)
    }
}

impl<DataIn: Data, DataOut: Data, DataRef: Data> ModelOp<DataIn, DataOut, DataRef, DataRef>
    for InputMappingOp<DataIn, DataOut>
{
    impl_model_op_for_input_transformation_op!(DataIn, DataOut, DataRef, DataRef);
}

pub struct ReferenceMappingOp<DataRefIn: Data, DataRefOut: Data> {
    f: fn(DataRefIn) -> DataRefOut,
    fp: fn(DataRefOut) -> DataRefIn,
}

impl<DataRefIn: Data, DataRefOut: Data> ReferenceMappingOp<DataRefIn, DataRefOut> {
    pub fn new(
        mapper: fn(DataRefIn) -> DataRefOut,
        mapper_reversed: fn(DataRefOut) -> DataRefIn,
    ) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
        }
    }
}

impl<DataRefIn: Data, DataRefOut: Data> Model for ReferenceMappingOp<DataRefIn, DataRefOut> {
    impl_model_no_params!();
}

impl<DataRefIn: Data, DataRefOut: Data> ReferenceTransformationOp<DataRefIn, DataRefOut>
    for ReferenceMappingOp<DataRefIn, DataRefOut>
{
    fn transform(&mut self, reference: DataRefIn) -> DataRefOut {
        (self.f)(reference)
    }

    fn revert(&mut self, reference: DataRefOut) -> DataRefIn {
        (self.fp)(reference)
    }
}

impl<D: Data, DataRefIn: Data, DataRefOut: Data> ModelOp<D, D, DataRefIn, DataRefOut>
    for ReferenceMappingOp<DataRefIn, DataRefOut>
{
    impl_model_op_for_reference_transformation_op!(D, D, DataRefIn, DataRefOut);
}

pub struct TotalMappingOp<DataIn: Data, DataOut: Data> {
    f: fn(DataIn) -> DataOut,
    fp: fn(DataOut) -> DataIn,
}

impl<DataIn: Data, DataOut: Data> TotalMappingOp<DataIn, DataOut> {
    pub fn new(mapper: fn(DataIn) -> DataOut, mapper_reversed: fn(DataOut) -> DataIn) -> Self {
        Self {
            f: mapper,
            fp: mapper_reversed,
        }
    }
}

impl<DataIn: Data, DataOut: Data> Model for TotalMappingOp<DataIn, DataOut> {
    impl_model_no_params!();
}

impl<DataIn: Data, DataOut: Data> TotalTransformationOp<DataIn, DataOut>
    for TotalMappingOp<DataIn, DataOut>
{
    fn transform(&mut self, input: DataIn) -> DataOut {
        (self.f)(input)
    }

    fn revert(&mut self, output: DataOut) -> DataIn {
        (self.fp)(output)
    }
}

impl<DataIn: Data, DataOut: Data> ModelOp<DataIn, DataOut, DataIn, DataOut>
    for TotalMappingOp<DataIn, DataOut>
{
    impl_model_op_for_total_transformation_op!(DataIn, DataOut, DataIn, DataOut);
}
