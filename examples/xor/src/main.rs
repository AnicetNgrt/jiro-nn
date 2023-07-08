use jiro_nn::linalg::{Matrix, MatrixTrait};

fn main() {
    let training_data = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0]
    ];

    let validation_data = training_data.clone();

    let training_tensor = Matrix::from_column_leading_vector2(&training_data);
}