use crate::linalg::{MatrixTrait, Matrix};
use super::ActivationLayer;

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.clone(), |m| Matrix::constant(m.dim().0, m.dim().1, 1.0))
}
