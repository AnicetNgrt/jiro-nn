use std::cmp::Ordering;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::linalg::{Matrix, MatrixTrait, Scalar};
use crate::network::NetworkLayer;
use crate::{activation::ActivationLayer, layer::dense_layer::DenseLayer, layer::Layer};

use super::{LearnableLayer, DropoutLayer, ParameterableLayer};

#[derive(Debug)]
pub struct FullLayer {
    dense: DenseLayer,
    activation: ActivationLayer,
    // dropout resources : https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    dropout_enabled: bool,
    dropout_rate: Option<Scalar>,
    mask: Option<Matrix>,
}

impl FullLayer {
    pub fn new(dense: DenseLayer, activation: ActivationLayer, dropout: Option<Scalar>) -> Self {
        Self {
            dense,
            activation,
            dropout_rate: dropout,
            dropout_enabled: false,
            mask: None
        }
    }
    
    fn generate_dropout_mask(
        &mut self,
        output_shape: (usize, usize),
    ) -> Option<(Matrix, Scalar)> {
        if let Some(dropout_rate) = self.dropout_rate {
            let mut rng = SmallRng::from_entropy();
            let dropout_mask = Matrix::from_fn(output_shape.0, output_shape.1, |_, _| {
                if rng
                    .gen_range((0.0 as Scalar)..(1.0 as Scalar))
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
                self.dense.weights = self.dense.weights.scalar_mul(1.0 - dropout_rate);
                let output = self.dense.forward(input);
                self.dense.weights = self.dense.weights.scalar_div(1.0 - dropout_rate);
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

impl NetworkLayer for FullLayer {}

impl ParameterableLayer for FullLayer {
    fn as_learnable_layer(&self) -> Option<&dyn LearnableLayer> {
        Some(self)
    }

    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn LearnableLayer> {
        Some(self)
    }

    fn as_dropout_layer(&mut self) -> Option<&mut dyn DropoutLayer> {
        Some(self)
    }
}

impl LearnableLayer for FullLayer {
    // returns a matrix of the (jxi) weights and the final column being the (j) biases
    fn get_learnable_parameters(&self) -> Vec<Vec<Scalar>> {
        let mut params = self.dense.weights.get_data();
        params.push(self.dense.biases.get_column(0));
        params
    }

    // takes a matrix of the (jxi) weights and the final column being the (j) biases
    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Scalar>>) {
        let mut weights = params_matrix.clone();
        let biases = weights.pop().unwrap();
        self.dense.weights = Matrix::from_column_leading_matrix(&weights);
        self.dense.biases = Matrix::from_column_vector(&biases);
    }
}

impl DropoutLayer for FullLayer {
    fn enable_dropout(&mut self) {
        self.dropout_enabled = true;
    }

    fn disable_dropout(&mut self) {
        self.dropout_enabled = false;
    }
}