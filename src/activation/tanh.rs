use crate::linalg::MatrixTrait;
use super::ActivationLayer;

pub fn hyperbolic_tangent(x: f64) -> f64 {
    let exp = libm::exp(x);
    let exp_neg = libm::exp(-x);
    (exp - exp_neg) / (exp + exp_neg)
}

pub fn hyperbolic_tangent_prime(x: f64) -> f64 {
    1. - libm::pow(hyperbolic_tangent(x), 2.)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(hyperbolic_tangent), |m| m.map(hyperbolic_tangent_prime))
}
