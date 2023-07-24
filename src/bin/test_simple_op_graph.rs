use jiro_nn::{
    linalg::{Matrix, MatrixTrait},
    ops::{op_graphs_builders::{
        op_graph_builder::{OpGraphBuilder, OpGraphBuilderAnchor},
        op_portal_builder::OpPortalBuilder,
    }, op_graphs::op_node::OpNodeTrait},
};


fn main() {
    let preprocessing = OpPortalBuilder::new(
        OpGraphBuilderAnchor::start()
            .vec2_to_matrix()
    );

    let op_build = OpGraphBuilder::start_from_data(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    )
    .portal(&preprocessing)
    // .dense(10)
    // .everything_adam_optimized()
    // .end()
    // .end()
    .tanh();

    let mut op = op_build.build_graph();
    let pred = op.run_inference();

    println!("pred: {:?}", pred);

    let pred = preprocessing
        .get_portal_to_op()
        .backward_or_revert(pred, Matrix::zeros(1, 1));

    println!("pred: {:?}", pred);
}
