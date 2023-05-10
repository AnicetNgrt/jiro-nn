use super::ActivationLayer;
use crate::linalg::{Matrix, MatrixTrait};

pub fn new() -> ActivationLayer {
    ActivationLayer::new(
        |m| m.maxof(&Matrix::constant(m.dim().0, m.dim().1, 0.)),
        |m| m.sign().maxof(&Matrix::constant(m.dim().0, m.dim().1, 0.)),
    )
}
