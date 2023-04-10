use nalgebra::{DMatrix};

use crate::activation::Activation;

pub mod dense_layer;
pub mod full_layer;

pub enum Layers {
    Dense,
    Activation(Activation),
}

pub trait Layer {
    // returns: j outputs
    fn forward(&mut self, input: DMatrix<f64>) -> DMatrix<f64>;

    // output_gradient: ∂E/∂Y
    // returns: ∂E/∂X
    fn backward(&mut self, epoch: usize, output_gradient: DMatrix<f64>)
        -> DMatrix<f64>;
}
