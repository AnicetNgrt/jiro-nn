use std::cmp::Ordering;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use nalgebra::{DMatrix};

use crate::{activation::ActivationLayer, layer::dense_layer::DenseLayer, layer::Layer};

pub struct FullLayer {
    dense: DenseLayer,
    activation: ActivationLayer,
    dropout_rate: Option<f64>,
    mask: Option<DMatrix<f64>>,
}

impl FullLayer {
    pub fn new(dense: DenseLayer, activation: ActivationLayer) -> Self {
        Self {
            dense,
            activation,
            dropout_rate: None,
            mask: None
        }
    }

    pub fn set_dropout_rate(&mut self, rate: f64) {
        self.dropout_rate = Some(rate);
    }

    pub fn remove_dropout_rate(&mut self) {
        self.dropout_rate = None;
    }

    pub fn update_dropout_rate(&mut self, f: impl Fn(f64) -> f64) {
        let rate = self.dropout_rate.unwrap();
        self.set_dropout_rate(f(rate));
    }

    fn generate_dropout_mask(
        &mut self,
        output_shape: (usize, usize),
    ) -> Option<(DMatrix<f64>, f64)> {
        if let Some(dropout_rate) = self.dropout_rate {
            let mut rng = SmallRng::from_entropy();
            let dropout_mask = DMatrix::from_fn(output_shape.0, output_shape.1, |_, _| {
                if rng
                    .gen_range(0.0f64..1.0f64)
                    .total_cmp(&self.dropout_rate.unwrap())
                    == Ordering::Greater
                {
                    1.0
                } else {
                    0.0
                }
            });
            Some((dropout_mask, dropout_rate))
        } else {
            None
        }
    }
}

impl Layer for FullLayer {
    fn forward(&mut self, mut input: DMatrix<f64>) -> DMatrix<f64> {
        
        if let Some((mask, dropout_rate)) = self.generate_dropout_mask(input.shape()) {
            input = input.component_mul(&mask) * (1.0 / (1.0 - dropout_rate));
            self.mask = Some(mask);
        }

        let output = self.dense.forward(input); 
        self.activation.forward(output)
    }

    fn backward(&mut self, epoch: usize, output_gradient: DMatrix<f64>) -> DMatrix<f64> {
        let activation_input_gradient = self.activation.backward(epoch, output_gradient);
        let input_gradient = self.dense
            .backward(epoch, activation_input_gradient);
    
        if let Some(mask) = &self.mask {
            input_gradient.component_mul(&mask)
        } else {
            input_gradient
        }
    }
}
