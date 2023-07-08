use jiro_nn::{linalg::{Matrix, MatrixTrait}, activation::Activation, layer::Layer, loss::Losses};

pub fn main() {
    let m = Matrix::from_column_leading_vec2(&vec![
        vec![1.0, 2.0, 3.0, 4.0, 0.1],
        vec![3.0, 2.0, 0.0, 2.0, 0.3],
        vec![6.0, 3.0, 1.0, 7.0, 0.4],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ]);

    m.print();

    let mut activation = Activation::Softmax.to_layer();
    let result = activation.forward(m.clone());
    
    result.print();

    let true_m = Matrix::from_column_leading_vec2(&vec![
        vec![0.0, 0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
    ]);

    true_m.print();

    let error = Losses::BCE.to_loss().loss_prime(&true_m, &result);

    error.print();

    let jacobian = activation.backward(0, error);

    jacobian.print();
}