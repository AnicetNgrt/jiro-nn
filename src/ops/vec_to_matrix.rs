use crate::{
    linalg::{Matrix, MatrixTrait, Scalar},
    ops::mappings::mapping_builders::TotalMappingBuilder,
};

use super::{
    mappings::mapping_builders::{InputMappingBuilder, ReferenceMappingBuilder},
    op_graphs_builders::{
        linkable_op_builder::LinkableOpBuilder, op_graph_builder::OpGraphBuilder,
    },
    Data,
};

// impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>> OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, DataRefOut>

impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>>
    OpGraphBuilder<'g, DataIn, Vec<Vec<Scalar>>, DataRefIn, Vec<Vec<Scalar>>>
{
    pub fn vec2_to_matrix(self) -> OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, Matrix> {
        let op_builder = TotalMappingBuilder::new(
            |vecs| Matrix::from_column_leading_vector2(&vecs),
            |matrix| matrix.get_data_col_leading(),
            |vec_meta| (vec_meta[0].len(), vec_meta.len()),
        );

        self.link_and_pack(op_builder)
    }
}

impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, Vec<Vec<Scalar>>, DataRefIn, DataRefOut>
{
    pub fn data_vec2_to_matrix(self) -> OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut> {
        let op_builder = InputMappingBuilder::new(
            |vecs| Matrix::from_column_leading_vector2(&vecs),
            |matrix| matrix.get_data_col_leading(),
            |vec_meta| (vec_meta[0].len(), vec_meta.len()),
        );

        self.link_and_pack(op_builder)
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRefIn: Data<'g>>
    OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, Vec<Vec<Scalar>>>
{
    pub fn ref_vec2_to_matrix(self) -> OpGraphBuilder<'g, DataIn, DataOut, DataRefIn, Matrix> {
        let op_builder = ReferenceMappingBuilder::new(
            |vecs| Matrix::from_column_leading_vector2(&vecs),
            |matrix| matrix.get_data_col_leading(),
            |vec_meta| (vec_meta[0].len(), vec_meta.len()),
        );

        self.link_and_pack(op_builder)
    }
}
