#![allow(unused_parens)]

use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    mappings::{
        input_mapping::{impl_op_builder_from_input_transformation_closures, InputMappingOp},
        reference_mapping::{
            impl_op_builder_from_reference_transformation_closures, ReferenceMappingOp,
        },
        total_mapping::{
            impl_op_builder_from_total_transformation_closures, TotalMappingOp,
        },
    },
    op_graphs::op_node::OpNodeTrait,
    op_graph_builder::{
        plug_builder_on_op_node_builder_data_out,
        plug_builder_on_op_node_builder_reference_out,
        plug_builder_on_op_node_builder_total_out, CombinatoryOpBuilder, OpGraphBuilder,
        OpNodeBuilder,
    },
    Data, 
};

pub struct InputVec2ToMatrixBuilder;

impl_op_builder_from_input_transformation_closures!(
    InputVec2ToMatrixBuilder,
    Vec<Vec<Scalar>>,
    Matrix,
    (|vecs| Matrix::from_column_leading_vector2(&vecs)),
    (|matrix| matrix.get_data_col_leading()),
    (|vec_meta| (vec_meta[0].len(), vec_meta.len()))
);

plug_builder_on_op_node_builder_data_out!(
    data_vec2_to_matrix,
    Vec<Vec<Scalar>>,
    Matrix,
    InputVec2ToMatrixBuilder
);

pub struct ReferenceVec2ToMatrixBuilder;

impl_op_builder_from_reference_transformation_closures!(
    ReferenceVec2ToMatrixBuilder,
    Vec<Vec<Scalar>>,
    Matrix,
    (|vecs| Matrix::from_column_leading_vector2(&vecs)),
    (|matrix| matrix.get_data_col_leading()),
    (|vec_meta| (vec_meta[0].len(), vec_meta.len()))
);

plug_builder_on_op_node_builder_reference_out!(
    reference_vec2_to_matrix,
    Vec<Vec<Scalar>>,
    Matrix,
    ReferenceVec2ToMatrixBuilder
);

pub struct Vec2ToMatrixBuilder;

impl_op_builder_from_total_transformation_closures!(
    Vec2ToMatrixBuilder,
    Vec<Vec<Scalar>>,
    Matrix,
    (|vecs| Matrix::from_column_leading_vector2(&vecs)),
    (|matrix| matrix.get_data_col_leading()),
    (|vec_meta| (vec_meta[0].len(), vec_meta.len()))
);

plug_builder_on_op_node_builder_total_out!(
    vec2_to_matrix,
    Vec<Vec<Scalar>>,
    Matrix,
    Vec2ToMatrixBuilder
);
