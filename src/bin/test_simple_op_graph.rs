use jiro_nn::ops::{
    op_graphs::op_node::OpNodeTrait,
    op_graphs_builders::op_graph_builder::{graph, graph_root},
};

fn main() {
    let preprocessing = graph()
        .vec2_to_matrix()
        .make_pointer();

    let op_build = graph_root(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    )
    .pointer(&preprocessing)
    .custom_node(
        graph()
            .dense(3)
            .end()
            .tanh()
    )
    // .dense(10)
    // .everything_adam_optimized()
    // .end()
    // .end()
    .tanh();

    let mut op = op_build.build_graph();
    let predictions = op.run_inference();

    println!("pred: {:?}", predictions);

    let predictions = preprocessing.get_op()
        .revert_reference(predictions);

    println!("pred: {:?}", predictions);
}
