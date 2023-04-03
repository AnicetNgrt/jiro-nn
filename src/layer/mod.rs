use nalgebra::SVector;

use crate::activation::Activation;

pub mod dense_layer;
pub mod full_layer;
pub mod hidden_layer;
pub mod skip_layer;

pub enum Layers {
    Dense,
    Activation(Activation)
}

pub trait Layer<const I: usize, const J: usize> {
    // returns: j outputs
    fn forward(&mut self, input: SVector<f64, I>) -> SVector<f64, J>;

    // output_gradient: ∂E/∂Y
    // returns: ∂E/∂X
    fn backward(&mut self, output_gradient: SVector<f64, J>, learning_rate: f64)
        -> SVector<f64, I>;
}
