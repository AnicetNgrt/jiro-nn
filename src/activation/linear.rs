use super::ActivationLayer;
use crate::linalg::{Matrix, MatrixTrait};

pub fn new() -> ActivationLayer {
    ActivationLayer::new(
        |m| m.clone(),
        |m| Matrix::constant(m.dim().0, m.dim().1, 1.0),
    )
}
