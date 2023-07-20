use super::ActivationLayer;
use crate::linalg::{Matrix, MatrixTrait};
use std::thread::available_parallelism;

// Formulas references from:
// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
// https://www.youtube.com/watch?v=AbLvJVwySEo

fn stablesoftmax_col(col: &Matrix) -> Matrix {
    let shiftx = col.scalar_sub(col.max());
    let exps = shiftx.exp();
    let sum = exps.sum();
    exps.scalar_div(sum)
}

fn stablesoftmax(m: &Matrix) -> Matrix {
    let ncol = m.dim().1;
    let mut columns: Vec<Matrix> = Vec::with_capacity(ncol);

    let n_threads = available_parallelism().unwrap().get().min(ncol / 10);
    if Matrix::is_backend_thread_safe() && n_threads > 1 {
        let step_size = (ncol as f32 / n_threads as f32).ceil() as usize;

        let mut threads = Vec::with_capacity(ncol);
        for i in (0..ncol).step_by(step_size) {
            let mut thread_columns = Vec::with_capacity(step_size);
            for j in i..(i + step_size).min(ncol) {
                let col = m.get_column_as_matrix(j);
                thread_columns.push(col);
            }
            threads.push(std::thread::spawn(move || {
                let mut results = Vec::with_capacity(step_size);
                for col in thread_columns {
                    let result = stablesoftmax_col(&col);
                    results.push(result);
                }
                results
            }));
        }
        for thread in threads {
            columns.extend(thread.join().unwrap());
        }
    } else {
        for i in 0..ncol {
            let col = m.get_column_as_matrix(i);
            let result = stablesoftmax_col(&col);
            columns.push(result);
        }
    }

    Matrix::from_column_matrices(&columns)
}

fn softmax_prime_col(col: &Matrix, output_gradient: &Matrix) -> Matrix {
    let n = col.dim().0;
    let ones = Matrix::constant(1, n, 1.0);
    let col_repeated = col.dot(&ones);
    let identity = Matrix::identity(n);
    let result = col_repeated
        .component_mul(&identity.component_sub(&col_repeated.transpose()))
        .dot(output_gradient);
    result
}

fn softmax_prime(m: &Matrix, output_gradient: &Matrix) -> Matrix {
    let ncol = m.dim().1;
    let mut columns: Vec<Matrix> = Vec::with_capacity(ncol);

    let n_threads = available_parallelism().unwrap().get().min(ncol / 10);
    if Matrix::is_backend_thread_safe() && n_threads > 1 {
        let step_size = (ncol as f32 / n_threads as f32).ceil() as usize;

        let mut threads = Vec::with_capacity(ncol);
        for i in (0..ncol).step_by(step_size) {
            let mut thread_columns = Vec::with_capacity(step_size);
            for j in i..(i + step_size).min(ncol) {
                let col = m.get_column_as_matrix(j);
                let grad_col = output_gradient.get_column_as_matrix(j);
                thread_columns.push((col, grad_col));
            }
            threads.push(std::thread::spawn(move || {
                let mut results = Vec::with_capacity(step_size);
                for (col, grad_col) in thread_columns {
                    let result = softmax_prime_col(&col, &grad_col);
                    results.push(result);
                }
                results
            }));
        }
        for thread in threads {
            columns.extend(thread.join().unwrap());
        }
    } else {
        for i in 0..ncol {
            let col = m.get_column_as_matrix(i);
            let grad_col = output_gradient.get_column_as_matrix(i);
            let result = softmax_prime_col(&col, &grad_col);
            columns.push(result);
        }
    }

    Matrix::from_column_matrices(&columns)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new_grad_dep(stablesoftmax, softmax_prime)
}
