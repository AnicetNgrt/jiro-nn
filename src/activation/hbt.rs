use crate::linalg::{MatrixTrait, Scalar};

use super::ActivationLayer;

fn hbt(x: Scalar) -> Scalar {
    let exp = libm::exp(x as f64) as Scalar;
    let exp_neg = libm::exp(-x as f64) as Scalar;
    (exp - exp_neg) / (exp + exp_neg)
}

fn hbt_prime(x: Scalar) -> Scalar {
    1. - libm::pow(hbt(x) as f64, 2.) as Scalar
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(hbt), |m| m.map(hbt_prime))
}
