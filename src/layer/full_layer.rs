use std::cmp::Ordering;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::linalg::{Matrix, MatrixTrait};
use crate::{activation::ActivationLayer, layer::dense_layer::DenseLayer, layer::Layer};

pub struct FullLayer {
    dense: DenseLayer,
    activation: ActivationLayer,
    // dropout resources : https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    dropout_enabled: bool,
    dropout_rate: Option<f64>,
    mask: Option<Matrix>,
}

impl FullLayer {
    pub fn new(dense: DenseLayer, activation: ActivationLayer, dropout: Option<f64>) -> Self {
        Self {
            dense,
            activation,
            dropout_rate: dropout,
            dropout_enabled: false,
            mask: None
        }
    }

    pub fn enable_dropout(&mut self) {
        self.dropout_enabled = true;
    }

    pub fn disable_dropout(&mut self) {
        self.dropout_enabled = false;
    }
    
    fn generate_dropout_mask(
        &mut self,
        output_shape: (usize, usize),
    ) -> Option<(Matrix, f64)> {
        if let Some(dropout_rate) = self.dropout_rate {
            let mut rng = SmallRng::from_entropy();
            let dropout_mask = Matrix::from_fn(output_shape.0, output_shape.1, |_, _| {
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
    fn forward(&mut self, mut input: Matrix) -> Matrix {
        let output = if self.dropout_enabled {
            if let Some((mask, _)) = self.generate_dropout_mask(input.dim()) {
                input = input.component_mul(&mask);
                self.mask = Some(mask);
            };
            self.dense.forward(input)
        } else {
            if let Some(dropout_rate) = self.dropout_rate {
                self.dense.map_weights(|w| w*(1.0-dropout_rate));
                let output = self.dense.forward(input);
                self.dense.map_weights(|w| w/(1.0-dropout_rate));
                output
            } else {
                self.dense.forward(input)
            }
        };

        self.activation.forward(output)
    }

    fn backward(&mut self, epoch: usize, output_gradient: Matrix) -> Matrix {
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
