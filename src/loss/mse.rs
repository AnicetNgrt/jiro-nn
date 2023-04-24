use crate::{loss::Loss, linalg::{Matrix, MatrixTrait, Scalar}};

pub fn mse_vec(y_pred: &Vec<Scalar>, y_true: &Vec<Scalar>) -> Scalar {
    let n_samples = y_pred.len();
    let mut sum = 0.0;
    for j in 0..n_samples {
        let diff = y_pred[j] - y_true[j];
        sum += diff * diff;
    }
    sum / ((n_samples) as Scalar)
}

fn mse(y_pred: &Matrix, y_true: &Matrix) -> Scalar {
    let n_samples = y_pred.dim().1;
    let n_outputs = y_pred.dim().0;

    let mut sum = 0.0;
    for j in 0..n_samples {
        for i in 0..n_outputs {
            let diff = y_pred.index(i, j) - y_true.index(i, j);
            sum += diff * diff;
        }
    }
    sum / ((n_samples * n_outputs) as Scalar)
}

fn mse_prime(y_pred: &Matrix, y_true: &Matrix) -> Matrix {
    let n_samples = y_pred.dim().1;
    let n_outputs = y_pred.dim().0;

    let mut mse_prime = Matrix::zeros(n_outputs, n_samples);
    for j in 0..n_samples {
        for i in 0..n_outputs {
            *mse_prime.index_mut(i, j) = -2.0 * (y_pred.index(i, j) - y_true.index(i, j));
        }
    }
    mse_prime
}

pub fn new() -> Loss {
    Loss::new(
        mse,
        mse_prime,
    )
}