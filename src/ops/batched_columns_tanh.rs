use crate::linalg::{Matrix, MatrixTrait};

use super::{
    batched_columns_activation::BatchedColumnsActivation,
    model_op_builder::{CombinatoryOpBuilder, OpBuild, OpBuilder},
    Data, ModelOp,
};

fn tanh(m: &Matrix) -> Matrix {
    let exp = m.exp();
    let exp_neg = m.scalar_mul(-1.).exp();
    (exp.component_sub(&exp_neg)).component_div(&(exp.component_add(&exp_neg)))
}

fn tanh_prime(m: &Matrix) -> Matrix {
    let hbt = tanh(m);
    let hbt2 = &hbt.square();
    let ones = Matrix::constant(hbt.dim().0, hbt.dim().1, 1.0);
    ones.component_sub(&hbt2)
}

pub fn batched_columns_tanh() -> BatchedColumnsActivation {
    BatchedColumnsActivation::new(tanh, tanh_prime)
}

pub struct TanhBuilder;

impl<DataRef: Data> OpBuilder<Matrix, Matrix, DataRef, DataRef> for TanhBuilder {
    fn build(
        &mut self,
        sample_data: Matrix,
        sample_ref: DataRef,
    ) -> (
        Box<dyn ModelOp<Matrix, Matrix, DataRef, DataRef>>,
        (Matrix, DataRef),
    ) {
        (Box::new(batched_columns_tanh()), (sample_data, sample_ref))
    }
}

impl<'a, DataIn: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuild<'a, DataIn, Matrix, DataRefIn, DataRefOut>
{
    pub fn tanh(self) -> OpBuild<'a, DataIn, Matrix, DataRefIn, DataRefOut> {
        self.push_and_pack(TanhBuilder)
    }
}
