use nalgebra::{DMatrix};

use crate::{layer::Layer, optimizer::{Optimizers, sgd::SGD}, initializers::Initializers, learning_rate::default_learning_rate};

pub struct DenseLayer {
    // i inputs, j outputs, i x j connections
    input: Option<DMatrix<f64>>,
    // j x i connection weights
    weights: DMatrix<f64>,
    // j output biases (single column)
    biases: DMatrix<f64>,
    weights_optimizer: Optimizers,
    biases_optimizer: Optimizers,
}

impl DenseLayer {
    pub fn new(
        i: usize,
        j: usize,
        weights_optimizer: Optimizers,
        biases_optimizer: Optimizers,
        weights_initializer: Initializers,
        biases_initializer: Initializers,
    ) -> Self {
        // about weights initialization : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        let weights = weights_initializer.gen_matrix(i, j);
        let biases = DMatrix::from_columns(&[biases_initializer.gen_vector(j)]);

        Self {
            weights: weights,
            biases: biases,
            input: None,
            weights_optimizer,
            biases_optimizer,
        }
    }

    pub fn map_weights(&mut self, f: impl Fn(f64) -> f64) {
        self.weights = self.weights.map(f);
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        // Y = W . X + B
        let mut res = &input * &self.weights.transpose();
        // add the biases to each row of res
        res.column_iter_mut()
            .zip(self.biases.iter())
            .for_each(|(mut col, bias)| {
                col.add_scalar_mut(*bias);
            });
        self.input = Some(input);
        // each row -> one batch of inputs for one neuron
        // each column -> inputs for each next neurons
        res
    }

    fn backward(&mut self, epoch: usize, output_gradient: DMatrix<f64>) -> DMatrix<f64> {
        let input = self.input.clone().unwrap();

        let weights_gradient = &output_gradient.transpose() * &input;
        
        let biases_gradient = DMatrix::from_columns(&[output_gradient.transpose().column_sum()]);

        let input_gradient = output_gradient * &self.weights;

        self.weights = self.weights_optimizer.update_parameters(epoch, &self.weights, &weights_gradient);
        self.biases = self.biases_optimizer.update_parameters(epoch, &self.biases, &biases_gradient);

        input_gradient
    }
}

pub fn default_biases_initializer() -> Initializers {
    Initializers::Zeros
}

pub fn default_weights_initializer() -> Initializers {
    Initializers::GlorotUniform
}

pub fn default_biases_optimizer() -> Optimizers {
    Optimizers::SGD(SGD::new(default_learning_rate()))
}

pub fn default_weights_optimizer() -> Optimizers {
    Optimizers::SGD(SGD::new(default_learning_rate()))
}