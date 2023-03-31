use nalgebra::{SVector, SMatrix};

use crate::{layer::Layer, loss::Loss};

pub struct Network<const IN: usize, const OUT: usize, L: Layer<IN, OUT>, E: Loss<OUT>> {
    // May be one or more layers inside
    // A layer is a layer as long as it implements the Layer trait
    layer: L,
    loss: E,
    learning_rate: f64
}

impl<const IN: usize, const OUT: usize, L: Layer<IN, OUT>, E: Loss<OUT>> Network<IN, OUT, L, E> {
    pub fn new(layer: L, loss: E, learning_rate: f64) -> Self {
        Self { layer, loss, learning_rate }
    }

    pub fn predict(&mut self, input: SVector<f64, IN>) -> SVector<f64, OUT> {
        self.layer.forward(input)
    }

    pub fn fit<const S: usize>(&mut self, x_train: SMatrix<f64, IN, S>, y_train: SMatrix<f64, OUT, S>, epochs: usize) {
        for e in 0..epochs {
            let mut error = 0.;
            for i in 0..S {
                let input = x_train.column(i).into();
                let pred = self.layer.forward(input);

                let y_true = y_train.column(i).into();
                error += self.loss.loss(y_true, pred);

                let error_gradient = self.loss.loss_prime(y_true, pred);
                self.layer.backward(error_gradient, self.learning_rate);
            }
            error /= S as f64;
            println!("epoch {}/{} error={}", e+1, epochs, error);
        }
    }
}
