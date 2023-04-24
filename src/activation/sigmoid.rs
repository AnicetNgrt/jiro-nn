use crate::linalg::{MatrixTrait, Scalar};
use super::ActivationLayer;

fn sigmoid(x: Scalar) -> Scalar {
    1. / (1. + libm::exp(-x as f64) as Scalar)
}

fn sigmoid_prime(x: Scalar) -> Scalar {
    let sigofx = sigmoid(x);
    sigofx * (1. - sigofx)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(sigmoid), |m| m.map(sigmoid_prime))
}
