use nalgebra::SVector;

use crate::{activation_layer::ActivationLayer, dense_layer::DenseLayer, layer::Layer};

pub struct FullLayer<const I: usize, const J: usize> {
    dense: DenseLayer<I, J>,
    activation: ActivationLayer<J>,
}

impl<const I: usize, const J: usize> FullLayer<I, J> {
    pub fn new(dense: DenseLayer<I, J>, activation: ActivationLayer<J>) -> Self {
        Self { dense, activation }
    }
}

impl<const I: usize, const J: usize> Layer<I, J> for FullLayer<I, J> {
    fn forward(&mut self, input: nalgebra::SVector<f64, I>) -> SVector<f64, J> {
        let output = self.dense.forward(input);
        self.activation.forward(output)
    }

    fn backward(
        &mut self,
        output_gradient: nalgebra::SVector<f64, J>,
        learning_rate: f64,
    ) -> SVector<f64, I> {
        let activation_input_gradient = self.activation.backward(output_gradient, learning_rate);
        self.dense
            .backward(activation_input_gradient, learning_rate)
    }
}
