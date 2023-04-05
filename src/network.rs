use nalgebra::{SVector};

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

    fn _predict(&mut self, input: SVector<f64, IN>) -> SVector<f64, OUT> {
        self.layer.forward(input)
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self._predict(SVector::from_iterator(input))
            .into_iter()
            .map(|x| *x)
            .collect()
    }

    pub fn predict_many(&mut self, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        inputs
            .into_iter()
            .map(|v| self.predict(v.clone()))
            .collect()
    }

    fn _train(
        &mut self,
        x_train: Vec<SVector<f64, IN>>,
        y_train: Vec<SVector<f64, OUT>>,
        learning_rate: f64,
        loss: &Loss<OUT>,
    ) -> f64 {
        let mut error = 0.;
        let mut i = 0;
        for (input, y_true) in x_train.into_iter().zip(y_train.into_iter()) {
            let pred = self.layer.forward(input);
            let e = loss.loss(y_true, pred);
            error += e;

            let error_gradient = loss.loss_prime(y_true, pred);
            self.layer.backward(error_gradient, learning_rate);
            i += 1;
        }
        error /= i as f64;
        error
    }

    pub fn train(
        &mut self,
        x_train: &Vec<Vec<f64>>,
        y_train: &Vec<Vec<f64>>,
        learning_rate: f64,
        loss: &Loss<OUT>,
    ) -> f64 {
        self._train(
            x_train
                .into_iter()
                .map(|col| SVector::<f64, IN>::from_iterator(col.clone().into_iter()))
                .collect(),
            y_train
                .into_iter()
                .map(|col| SVector::<f64, OUT>::from_iterator(col.clone().into_iter()))
                .collect(),
            learning_rate,
            loss,
        )
    }
}
