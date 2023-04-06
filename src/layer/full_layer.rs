use std::{cell::RefCell, rc::Rc};

use nalgebra::SVector;

use crate::{activation::ActivationLayer, layer::dense_layer::DenseLayer, layer::Layer};

pub struct FullLayerConfig {
    dropout_rate: Rc<RefCell<Option<f64>>>
}

impl FullLayerConfig {
    pub fn new() -> Self {
        Self {
            dropout_rate: Rc::new(RefCell::new(None))
        }
    }

    pub fn set_dropout_rate(&self, rate: f64) {
        *self.dropout_rate.borrow_mut() = Some(rate);
    }

    pub fn remove_dropout_rate(&self) {
        *self.dropout_rate.borrow_mut() = None;
    }

    pub fn update_dropout_rate(&self, f: impl Fn(f64) -> f64) {
        let rate = self.dropout_rate.borrow().unwrap();
        self.set_dropout_rate(f(rate));
    }
}

pub struct FullLayer<const I: usize, const J: usize> {
    dense: DenseLayer<I, J>,
    activation: ActivationLayer<J>,
    dropout_rate: Rc<RefCell<Option<f64>>>,
}

impl<const I: usize, const J: usize> FullLayer<I, J> {
    pub fn new(dense: DenseLayer<I, J>, activation: ActivationLayer<J>) -> Self {
        Self {
            dense,
            activation,
            dropout_rate: Rc::new(RefCell::new(None)),
        }
    }

    pub fn get_config(&self) -> FullLayerConfig {
        FullLayerConfig {
            dropout_rate: self.dropout_rate.clone()
        }
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
