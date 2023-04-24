use crate::linalg::{MatrixTrait, Scalar};
use super::ActivationLayer;

fn relu(x: Scalar) -> Scalar {
    x.max(0.)
}

fn relu_prime(x: Scalar) -> Scalar {
    if x > 0. {
        1.
    } else {
        0.
    }
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(relu), |m| m.map(relu_prime))
}
