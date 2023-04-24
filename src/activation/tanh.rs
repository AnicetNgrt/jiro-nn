use crate::linalg::{MatrixTrait, Scalar};
use super::ActivationLayer;

pub fn hyperbolic_tangent(x: Scalar) -> Scalar {
    let exp = libm::exp(x as f64) as Scalar;
    let exp_neg = libm::exp(-x as f64) as Scalar;
    (exp - exp_neg) / (exp + exp_neg)
}

pub fn hyperbolic_tangent_prime(x: Scalar) -> Scalar {
    1. - libm::pow(hyperbolic_tangent(x) as f64, 2.) as Scalar
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(hyperbolic_tangent), |m| m.map(hyperbolic_tangent_prime))
}
