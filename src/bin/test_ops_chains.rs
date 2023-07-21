use jiro_nn::{linalg::Matrix, ops::model_op_builder::OpGraphBuilder};

fn model() -> Matrix {
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

    pred
}

fn main() {
    let pred = model();
    println!("pred: {:?}", pred);
}
