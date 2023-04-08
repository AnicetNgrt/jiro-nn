use nalgebra::{DVector, DMatrix};

use crate::{layer::{full_layer::FullLayer, Layer}, loss::Loss};

pub struct Network {
    // May be one or more layers inside
    // A layer is a layer as long as it implements the Layer trait
    layers: Vec<FullLayer>,
    i: usize,
    j: usize
}

impl Network {
    pub fn new(layers: Vec<FullLayer>, i: usize, j: usize) -> Self {
        Self { layers, i, j }
    }

    fn _predict(&mut self, input: DVector<f64>) -> DVector<f64> {
        self.layers.forward(DMatrix::from_rows(&[input.transpose()])).row(0).transpose()
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self._predict(DVector::from_iterator(self.i, input))
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
        x_train: Vec<DVector<f64>>,
        y_train: Vec<DVector<f64>>,
        learning_rate: f64,
        loss: &Loss,
        batch_size: usize,
    ) -> f64 {
        let mut error = 0.;
        let mut i = 0;
        let transposed_x_train: Vec<_> = x_train.into_iter().map(|v| v.transpose()).collect();
        let transposed_y_train: Vec<_> = y_train.into_iter().map(|v| v.transpose()).collect();
        let x_train_batches: Vec<_> = transposed_x_train.chunks(batch_size).map(|c| c.to_vec()).collect();
        let y_train_batches: Vec<_> = transposed_y_train.chunks(batch_size).map(|c| c.to_vec()).collect();

        for (input_batch, y_true_batch) in x_train_batches.into_iter().zip(y_train_batches.into_iter()) {
            let pred = self.layers.forward(DMatrix::from_rows(input_batch.as_slice()));
            let y_true_batch_matrix = DMatrix::from_rows(y_true_batch.as_slice());
            let e = loss.loss(&y_true_batch_matrix, &pred);
            error += e;

            let error_gradient = loss.loss_prime(&y_true_batch_matrix, &pred);
            self.layers.backward(error_gradient, learning_rate);
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
        loss: &Loss,
        batch_size: usize,
    ) -> f64 {
        self._train(
            x_train
                .into_iter()
                .map(|col| DVector::<f64>::from_iterator(self.i, col.clone().into_iter()))
                .collect(),
            y_train
                .into_iter()
                .map(|col| DVector::<f64>::from_iterator(self.j, col.clone().into_iter()))
                .collect(),
            learning_rate,
            loss,
            batch_size
        )
    }

    pub fn set_dropout_rates(&mut self, rates: &Vec<f64>) {
        for (i, rate) in rates.into_iter().enumerate() {
            self.layers[i].get_config().set_dropout_rate(*rate);
        }
    }

    pub fn remove_dropout_rates(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.get_config().remove_dropout_rate();
        }
    }
}

impl<L: Layer> Layer for Vec<L> {
    fn forward(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        let mut output = input;
        for layer in self.iter_mut() {
            output = layer.forward(output);
        }
        output
    }

    fn backward(&mut self, error_gradient: DMatrix<f64>, learning_rate: f64) -> DMatrix<f64> {
        let mut error_gradient = error_gradient;
        for layer in self.iter_mut().rev() {
            error_gradient = layer.backward(error_gradient, learning_rate);
        }
        error_gradient
    }
}