use jiro_nn::ops::model_op_builder::OpGraphBuilder;

fn main() {
    let mut op_build = OpGraphBuilder::data_as_entry_point(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    )
    .vec2_to_matrix()
    .tanh();

    let mut op = op_build.build_graph();
    let pred = op.run_inference();
    
    println!("pred: {:?}", pred);
}
