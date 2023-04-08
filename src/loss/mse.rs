use nalgebra::{DMatrix};

use crate::loss::Loss;

pub fn mse_vecf64(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
    let n_samples = y_pred.len();
    let mut sum = 0.0;
    for j in 0..n_samples {
        let diff = y_pred[j] - y_true[j];
        sum += diff * diff;
    }
    sum / ((n_samples) as f64)
}

fn mse(y_pred: &DMatrix<f64>, y_true: &DMatrix<f64>) -> f64 {
    let n_samples = y_pred.ncols();
    let n_outputs = y_pred.nrows();

    let mut sum = 0.0;
    for j in 0..n_samples {
        for i in 0..n_outputs {
            let diff = y_pred[(i, j)] - y_true[(i, j)];
            sum += diff * diff;
        }
    }
    sum / ((n_samples * n_outputs) as f64)
}

fn mse_prime(y_pred: &DMatrix<f64>, y_true: &DMatrix<f64>) -> DMatrix<f64> {
    let n_samples = y_pred.ncols();
    let n_outputs = y_pred.nrows();

    let mut mse_prime = DMatrix::zeros(n_outputs, n_samples);
    for j in 0..n_samples {
        for i in 0..n_outputs {
            mse_prime[(i, j)] = -2.0 * (y_pred[(i, j)] - y_true[(i, j)]);
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