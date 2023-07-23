use crate::linalg::{Matrix, MatrixTrait};

use super::{
    batched_columns_activation::BatchedColumnsActivation,
    op_graphs::op_node::OpNodeTrait,
    op_graphs_builders::{
        linkable_op_builder::LinkableOpBuilder, op_graph_builder::OpGraphBuilder,
        op_node_builder::OpNodeBuilder,
    },
    Data,
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

impl<'g, DataRef: Data<'g>> OpNodeBuilder<'g, Matrix, Matrix, DataRef, DataRef> for TanhBuilder {
    fn build(
        &mut self,
        meta_data: (usize, usize),
        meta_ref: DataRef::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, Matrix, Matrix, DataRef, DataRef> + 'g>,
        ((usize, usize), DataRef::Meta),
    ) {
        (Box::new(batched_columns_tanh()), (meta_data, meta_ref))
    }
}

impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut>
{
    pub fn tanh(self) -> OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut> {
        self.link_and_pack(TanhBuilder)
    }
}
