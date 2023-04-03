use nalgebra::{SMatrix, SVector};

use crate::{layer::Layer, loss::Loss};

pub struct Network<const IN: usize, const OUT: usize> {
    // May be one or more layers inside
    // A layer is a layer as long as it implements the Layer trait
    layer: Box<dyn Layer<IN, OUT>>,
}

impl<const IN: usize, const OUT: usize> Network<IN, OUT> {
    pub fn new(layer: Box<dyn Layer<IN, OUT>>) -> Self {
        Self { layer }
    }

    pub fn predict(&mut self, input: SVector<f64, IN>) -> SVector<f64, OUT> {
        self.layer.forward(input)
    }

    pub fn predict_iter<I>(&mut self, input: I) -> Vec<f64>
    where
        I: IntoIterator<Item = f64>,
    {
        self.predict(SVector::from_iterator(input))
            .iter()
            .map(|x| *x)
            .collect()
    }

    pub fn train<const S: usize>(
        &mut self,
        x_train: SMatrix<f64, IN, S>,
        y_train: SMatrix<f64, OUT, S>,
        learning_rate: f64,
        loss: &Loss<OUT>,
    ) -> f64 {
        let mut error = 0.;
        for i in 0..S {
            let input = x_train.column(i).into();
            let pred = self.layer.forward(input);

            let y_true = y_train.column(i).into();
            let e = loss.loss(y_true, pred);
            error += e;

            let error_gradient = loss.loss_prime(y_true, pred);
            self.layer.backward(error_gradient, learning_rate);
        }
        error /= S as f64;
        error
    }

    pub fn train_iter<const S: usize, I, J>(
        &mut self,
        x_train: I,
        y_train: J,
        learning_rate: f64,
        loss: &Loss<OUT>,
    ) -> f64
    where
        I: IntoIterator<Item = f64>,
        J: IntoIterator<Item = f64>
    {
        self.train::<S>(
            SMatrix::from_iterator(x_train.into_iter()),
            SMatrix::from_iterator(y_train.into_iter()),
            learning_rate,
            loss,
        )
    }
}
