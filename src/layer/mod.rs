use nalgebra::{DVector};

use crate::activation::Activation;

pub mod dense_layer;
pub mod full_layer;

pub enum Layers {
    Dense,
    Activation(Activation),
}

pub trait Layer {
    // returns: j outputs
    fn forward(&mut self, input: DVector<f64>) -> DVector<f64>;

    // output_gradient: ∂E/∂Y
    // returns: ∂E/∂X
    fn backward(&mut self, output_gradient: DVector<f64>, learning_rate: f64)
        -> DVector<f64>;
}
