use crate::linalg::MatrixTrait;
use super::ActivationLayer;

fn linear(x: f64) -> f64 {
    x
}

fn linear_prime(_: f64) -> f64 {
    1.0
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(linear), |m| m.map(linear_prime))
}
