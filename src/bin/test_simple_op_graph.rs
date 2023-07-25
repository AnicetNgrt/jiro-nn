use jiro_nn::ops::{
    op_graphs::op_node::OpNodeTrait,
    op_graphs_builders::op_graph_builder::{OpGraphBuilder, OpGraphBuilderAnchor},
};

fn main() {
    let preprocessing = OpGraphBuilderAnchor::start().vec2_to_matrix().make_pointer();

    let op_build = OpGraphBuilder::start_from_data(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    )
    .pointer(&preprocessing)
    // .dense(10)
    // .everything_adam_optimized()
    // .end()
    // .end()
    .tanh();

    let mut op = op_build.build_graph();
    let predictions = op.run_inference();

    println!("pred: {:?}", predictions);

    let predictions = preprocessing
        .get_pointer_to_op()
        .revert_reference(predictions);

    println!("pred: {:?}", predictions);
}
