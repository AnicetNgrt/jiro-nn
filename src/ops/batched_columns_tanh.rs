use crate::linalg::{Matrix, MatrixTrait};

use super::{
    batched_columns_activation::BatchedColumnsActivation,
    op_graph_builder::{CombinatoryOpBuilder, OpGraphBuilder, OpSubgraphBuilder},
    Data, OpSubgraphTrait,
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

impl<'g, DataRef: Data<'g>>
    OpSubgraphBuilder<'g, Matrix, Matrix, DataRef, DataRef> for TanhBuilder
{
    fn build(
        &mut self,
        sample_data: Matrix,
        sample_ref: DataRef,
    ) -> (
        Box<dyn OpSubgraphTrait<'g, Matrix, Matrix, DataRef, DataRef> + 'g>,
        (Matrix, DataRef),
    ) {
        (Box::new(batched_columns_tanh()), (sample_data, sample_ref))
    }
}

impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut>
{
    pub fn tanh(self) -> OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut> {
        self.push_and_pack(TanhBuilder)
    }
}
