use nalgebra::{SVector, SMatrix};

use crate::{layer::Layer, loss::Loss};

pub struct Network<const IN: usize, const OUT: usize> {
    // May be one or more layers inside
    // A layer is a layer as long as it implements the Layer trait
    layer: Box<dyn Layer<IN, OUT>>
}

impl<const IN: usize, const OUT: usize> Network<IN, OUT> {
    pub fn new(layer: Box<dyn Layer<IN, OUT>>) -> Self {
        Self { layer }
    }

    pub fn predict(&mut self, input: SVector<f64, IN>) -> SVector<f64, OUT> {
        self.layer.forward(input)
    }

    pub fn fit<const S: usize, E: Loss<OUT>>(&mut self, x_train: SMatrix<f64, IN, S>, y_train: SMatrix<f64, OUT, S>, epochs: usize,  learning_rate: f64) {
        for e in 0..epochs {
            let mut error = 0.;
            for i in 0..S {
                let input = x_train.column(i).into();
                let pred = self.layer.forward(input);

                let y_true = y_train.column(i).into();
                error += E::loss(y_true, pred);

                let error_gradient = E::loss_prime(y_true, pred);
                self.layer.backward(error_gradient, learning_rate);
            }
            error /= S as f64;
            println!("epoch {}/{} error={}", e+1, epochs, error);
        }
    }
}
