use crate::linalg::{MatrixTrait, Scalar};
use super::ActivationLayer;

fn linear(x: Scalar) -> Scalar {
    x
}

fn linear_prime(_: Scalar) -> Scalar {
    1.0
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(linear), |m| m.map(linear_prime))
}
