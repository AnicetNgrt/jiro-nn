use jiro_nn::ops::model_op_builder::OpBuild;

fn main() {
    let mut op_build = OpBuild::from_data(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![
            vec![0.0], 
            vec![1.0], 
            vec![1.0], 
            vec![0.0]
        ],
    )
    .vec2_to_matrix()
    .tanh();

    let mut op = op_build.build_full_graph();
    let pred = op.run_inference();
}
