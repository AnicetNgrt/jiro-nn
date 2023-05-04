use crate::linalg::{MatrixTrait, Matrix};
use super::ActivationLayer;

fn tanh(m: &Matrix) -> Matrix {
    let exp = m.exp();
    let exp_neg = m.scalar_mul(-1.).exp();
    (exp.component_sub(&exp_neg)).component_div(&(exp.component_add(&exp_neg)))
}

fn tanh_prime(m: &Matrix) -> Matrix {
    let hbt = tanh(m);
    let hbt2 = &hbt.square();
    let ones = Matrix::constant(hbt.dim().0, hbt.dim().1, 1.0);
    ones.component_sub(&hbt2)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(tanh, tanh_prime)
}
