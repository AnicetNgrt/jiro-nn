use std::{rc::Rc, cell::RefCell};

use nalgebra::{SVector};

use crate::{activation::ActivationLayer, layer::dense_layer::DenseLayer, layer::Layer};

pub struct FullLayer<const I: usize, const J: usize> {
    dense: DenseLayer<I, J>,
    activation: ActivationLayer<J>,
    dropout_rate: Rc<RefCell<Option<f64>>>
}

impl<const I: usize, const J: usize> FullLayer<I, J> {
    pub fn new(dense: DenseLayer<I, J>, activation: ActivationLayer<J>) -> Self {
        Self { dense, activation, dropout_rate: Rc::new(RefCell::new(None)) }
    }

    pub fn access_dropout_rate(&self) -> Rc<RefCell<Option<f64>>> {
        self.dropout_rate.clone()
    }
}

impl<const I: usize, const J: usize> Layer<I, J> for FullLayer<I, J> {
    fn forward(&mut self, input: nalgebra::SVector<f64, I>) -> SVector<f64, J> {
        let input = if let Some(rate) = *self.dropout_rate.borrow() {
            SVector::<f64, I>::new_random()
                .map(|r| if r <= rate { 0. } else { 1. })
                .component_mul(&input)
        } else {
            input
        };

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
