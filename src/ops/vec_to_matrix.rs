#![allow(unused_parens)]

use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    mapping::{
        impl_op_builder_from_input_transformation_closures,
        impl_op_builder_from_reference_transformation_closures,
        impl_op_builder_from_total_transformation_closures, InputMappingOp, ReferenceMappingOp,
        TotalMappingOp,
    },
    op_graph_builder::{
        plug_builder_on_op_subgraph_builder_data_out, plug_builder_on_op_subgraph_builder_reference_out,
        plug_builder_on_op_subgraph_builder_total_out, CombinatoryOpBuilder, OpGraphBuilder, OpSubgraphBuilder,
    },
    Data, 
    op_graph::OpSubgraphTrait,
};

pub struct InputVec2ToMatrixBuilder;

impl_op_builder_from_input_transformation_closures!(
    InputVec2ToMatrixBuilder,
    Vec<Vec<Scalar>>,
    Matrix,
    (|vecs| Matrix::from_column_leading_vector2(&vecs)),
    (|matrix| matrix.get_data_col_leading())
);

plug_builder_on_op_subgraph_builder_data_out!(
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
    (|matrix| matrix.get_data_col_leading())
);

plug_builder_on_op_subgraph_builder_reference_out!(
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
    (|matrix| matrix.get_data_col_leading())
);

plug_builder_on_op_subgraph_builder_total_out!(
    vec2_to_matrix,
    Vec<Vec<Scalar>>,
    Matrix,
    Vec2ToMatrixBuilder
);
