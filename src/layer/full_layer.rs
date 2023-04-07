use std::{cell::RefCell, rc::Rc};

use nalgebra::{DVector};

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

pub struct FullLayer {
    dense: DenseLayer,
    activation: ActivationLayer,
    dropout_rate: Rc<RefCell<Option<f64>>>,
}

impl FullLayer {
    pub fn new(dense: DenseLayer, activation: ActivationLayer) -> Self {
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

impl Layer for FullLayer {
    fn forward(&mut self, input: DVector<f64>) -> DVector<f64> {
        let input = if let Some(rate) = *self.dropout_rate.borrow() {
            DVector::<f64>::new_random(input.nrows())
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
        output_gradient: DVector<f64>,
        learning_rate: f64,
    ) -> DVector<f64> {
        let activation_input_gradient = self.activation.backward(output_gradient, learning_rate);
        self.dense
            .backward(activation_input_gradient, learning_rate)
    }
}
