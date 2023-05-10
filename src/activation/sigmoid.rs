use super::ActivationLayer;
use crate::linalg::{Matrix, MatrixTrait};

fn sigmoid(m: &Matrix) -> Matrix {
    let exp_neg = m.scalar_mul(-1.).exp();
    let ones = Matrix::constant(m.dim().0, m.dim().1, 1.0);
    ones.component_div(&(ones.component_add(&exp_neg)))
}

fn sigmoid_prime(m: &Matrix) -> Matrix {
    let sig = sigmoid(m);
    let ones = Matrix::constant(sig.dim().0, sig.dim().1, 1.0);
    sig.component_mul(&(ones.component_sub(&sig)))
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(sigmoid, sigmoid_prime)
}
