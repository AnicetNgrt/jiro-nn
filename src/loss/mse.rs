use crate::{
    linalg::{Matrix, MatrixTrait, Scalar},
    loss::Loss,
};

pub fn mse_vec(y_true: &Vec<Scalar>, y_pred: &Vec<Scalar>) -> Scalar {
    let n_samples = y_pred.len();
    let mut sum = 0.0;
    for j in 0..n_samples {
        let diff = y_pred[j] - y_true[j];
        sum += diff * diff;
    }
    sum / ((n_samples) as Scalar)
}

fn mse(y_pred: &Matrix, y_true: &Matrix) -> Scalar {
    ((y_pred.component_sub(&y_true)).square()).mean()
}

fn mse_prime(y_pred: &Matrix, y_true: &Matrix) -> Matrix {
    (y_pred.component_sub(&y_true)).scalar_mul(-2.0)
}

pub fn new() -> Loss {
    Loss::new(mse, mse_prime)
}
