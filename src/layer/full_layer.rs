use std::cmp::Ordering;
use std::{cell::RefCell, rc::Rc};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use nalgebra::{DMatrix};

use crate::{activation::ActivationLayer, layer::dense_layer::DenseLayer, layer::Layer};

pub struct FullLayerConfig {
    dropout_rate: Rc<RefCell<Option<f64>>>,
}

impl FullLayerConfig {
    pub fn new() -> Self {
        Self {
            dropout_rate: Rc::new(RefCell::new(None)),
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
    mask: Option<DMatrix<f64>>,
}

impl FullLayer {
    pub fn new(dense: DenseLayer, activation: ActivationLayer) -> Self {
        Self {
            dense,
            activation,
            dropout_rate: Rc::new(RefCell::new(None)),
            mask: None
        }
    }

    pub fn get_config(&self) -> FullLayerConfig {
        FullLayerConfig {
            dropout_rate: self.dropout_rate.clone(),
        }
    }

    fn generate_dropout_mask(
        &mut self,
        output_shape: (usize, usize),
    ) -> Option<(DMatrix<f64>, f64)> {
        if let Some(dropout_rate) = *self.dropout_rate.borrow() {
            let mut rng = SmallRng::from_entropy();
            let dropout_mask = DMatrix::from_fn(output_shape.0, output_shape.1, |_, _| {
                if rng
                    .gen_range(0.0f64..1.0f64)
                    .total_cmp(&self.dropout_rate.borrow().unwrap())
                    == Ordering::Greater
                {
                    0.0
                } else {
                    1.0
                }
            });
            Some((dropout_mask, dropout_rate))
        } else {
            None
        }
    }
}

impl Layer for FullLayer {
    fn forward(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        let output = self.dense.forward(input);

        if let Some((mask, dropout_rate)) = self.generate_dropout_mask(output.shape()) {
            let res = self.activation
                .forward(output.component_mul(&mask) * (1.0 / (1.0 - dropout_rate)));
            self.mask = Some(mask);
            return res;
        }

        self.activation.forward(output)
    }

    fn backward(&mut self, epoch: usize, output_gradient: DMatrix<f64>) -> DMatrix<f64> {
        let activation_input_gradient = self.activation.backward(epoch, output_gradient);
        
        let activation_input_gradient = if let Some(mask) = &self.mask {
            activation_input_gradient.component_mul(&mask)
        } else {
            activation_input_gradient
        };

        self.dense
            .backward(epoch, activation_input_gradient)
    }
}
