use crate::{
    linalg::{Matrix, MatrixTrait, Scalar},
    loss::Loss,
};

pub fn bce_vec(y_pred: &Vec<Scalar>, y_true: &Vec<Scalar>) -> Scalar {
    let n_samples = y_pred.len();
    let mut sum = 0.0;
    for j in 0..n_samples {
        sum += y_pred[j] * y_true[j].ln() + (1.0 - y_pred[j]) * (1.0 - y_true[j]).ln();
    }
    sum / ((n_samples) as Scalar)
}

fn bce(y_pred: &Matrix, y_true: &Matrix) -> Scalar {
    let ones = Matrix::constant(y_true.dim().0, y_true.dim().1, 1.);
    (y_true.component_mul(&y_pred.log()).component_add(
        &ones
            .component_sub(&y_true)
            .component_mul(&ones.component_sub(&y_pred).log()),
    ))
    .mean()
}

fn bce_prime(y_pred: &Matrix, y_true: &Matrix) -> Matrix {
    let ones = Matrix::constant(y_true.dim().0, y_true.dim().1, 1.);
    let ones_m_yt = ones.component_sub(&y_true);
    let ones_m_yp = ones.component_sub(&y_pred);

    ones_m_yt
        .component_div(&ones_m_yp)
        .component_sub(&y_true.component_div(y_pred))
        .scalar_div(y_pred.dim().0 as Scalar)
}

pub fn new() -> Loss {
    Loss::new(bce, bce_prime)
}
