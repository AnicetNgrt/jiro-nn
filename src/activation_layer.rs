use nalgebra::SVector;

use crate::layer::Layer;

pub type ActivationFn<const I: usize> = fn(&SVector<f64, I>) -> SVector<f64, I>;

pub struct ActivationLayer<const I: usize> {
    // i inputs = i outputs (it's just a map)
    input: Option<SVector<f64, I>>,
    activation: ActivationFn<I>,
    derivative: ActivationFn<I>,
}

impl<const I: usize> ActivationLayer<I> {
    pub fn new(activation: ActivationFn<I>, derivative: ActivationFn<I>) -> Self {
        Self {
            input: None,
            activation,
            derivative,
        }
    }
}

impl<const I: usize> Layer<I, I> for ActivationLayer<I> {
    fn forward(&mut self, input: SVector<f64, I>) -> SVector<f64, I> {
        self.input = Some(input);
        (self.activation)(&input)
    }

    fn backward(
        &mut self,
        output_gradient: SVector<f64, I>,
        _learning_rate: f64,
    ) -> SVector<f64, I> {
        // ∂E/∂X = ∂E/∂Y ⊙ f'(X)
        let fprime_x = (self.derivative)(&self.input.unwrap());
        output_gradient.component_mul(&fprime_x)
    }
}
