use nalgebra::{DMatrix, DVector};

use crate::{layer::Layer, optimizer::Optimizers};

pub struct DenseLayer {
    // i inputs, j outputs, i x j connections
    input: Option<DMatrix<f64>>,
    // j x i connection weights
    weights: DMatrix<f64>,
    // j output biases
    biases: DVector<f64>,
    optimizer: Optimizers,
}

impl DenseLayer {
    pub fn new(
        i: usize,
        j: usize,
        optimizer: Optimizers,
        initial_weights_range: (f64, f64),
    ) -> Self {
        let weights = DMatrix::<f64>::new_random(j, i);
        let biases = DVector::<f64>::new_random(j);

        let weights = (weights * (initial_weights_range.1 - initial_weights_range.0))
            .add_scalar(initial_weights_range.0);

        Self {
            weights: weights,
            biases: biases.add_scalar(-0.5) * 2.,
            input: None,
            optimizer,
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

        let biases_gradient = &output_gradient.transpose().column_sum();

        let input_gradient = output_gradient * &self.weights;

        let lr = self.optimizer.update_learning_rate(epoch);
        self.weights = self
            .optimizer
            .update_weights(epoch, &self.weights, &weights_gradient);
        self.biases = &self.biases - (lr * biases_gradient);

        input_gradient
    }
}
