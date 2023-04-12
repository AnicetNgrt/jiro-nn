use nalgebra::{DMatrix, DVector};

use crate::{
    layer::{full_layer::FullLayer, Layer},
    loss::Loss,
};

pub struct Network {
    // May be one or more layers inside
    // A layer is a layer as long as it implements the Layer trait
    layers: Vec<FullLayer>,
    pub i: usize,
    pub j: usize,
}

impl Network {
    pub fn new(layers: Vec<FullLayer>, i: usize, j: usize) -> Self {
        Self { layers, i, j }
    }

    fn _predict(&mut self, input: DVector<f64>) -> DVector<f64> {
        self.layers
            .forward(DMatrix::from_rows(&[input.transpose()]))
            .row(0)
            .transpose()
    }

    pub fn predict_evaluate(
        &mut self,
        input: Vec<f64>,
        y: Vec<f64>,
        loss: &Loss,
    ) -> (Vec<f64>, f64) {
        let preds: Vec<_> = self
            ._predict(DVector::from_iterator(self.i, input))
            .into_iter()
            .map(|x| *x)
            .collect();

        let loss = loss.loss_vec(&vec![y], &vec![preds.clone()]);

        (preds, loss)
    }

    pub fn predict_evaluate_many(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        ys: &Vec<Vec<f64>>,
        loss: &Loss,
    ) -> (Vec<Vec<f64>>, f64, f64) {
        let preds_losses: Vec<_> = inputs
            .into_iter()
            .zip(ys.into_iter())
            .map(|(x, y)| self.predict_evaluate(x.clone(), y.clone(), loss))
            .collect();

        let preds = preds_losses.iter().map(|(p, _)| p.clone()).collect();
        let losses: Vec<_> = preds_losses.iter().map(|(_, l)| *l).collect();
        let avg_loss = losses.iter().sum::<f64>() / preds_losses.len() as f64;
        let std_loss = losses
            .iter()
            .fold(0., |acc, x| acc + (x - avg_loss).powi(2))
            / preds_losses.len() as f64;

        (preds, avg_loss, std_loss)
    }

    fn _train(
        &mut self,
        epoch: usize,
        x_train: Vec<DVector<f64>>,
        y_train: Vec<DVector<f64>>,
        loss: &Loss,
        batch_size: usize,
    ) -> f64 {
        let mut error = 0.;
        let mut i = 0;
        let transposed_x_train: Vec<_> = x_train.into_iter().map(|v| v.transpose()).collect();
        let transposed_y_train: Vec<_> = y_train.into_iter().map(|v| v.transpose()).collect();
        let x_train_batches: Vec<_> = transposed_x_train
            .chunks(batch_size)
            .map(|c| c.to_vec())
            .collect();
        let y_train_batches: Vec<_> = transposed_y_train
            .chunks(batch_size)
            .map(|c| c.to_vec())
            .collect();

        for (input_batch, y_true_batch) in
            x_train_batches.into_iter().zip(y_train_batches.into_iter())
        {
            let pred = self
                .layers
                .forward(DMatrix::from_rows(input_batch.as_slice()));
            let y_true_batch_matrix = DMatrix::from_rows(y_true_batch.as_slice());
            let e = loss.loss(&y_true_batch_matrix, &pred);
            error += e;

            let error_gradient = loss.loss_prime(&y_true_batch_matrix, &pred);
            self.layers.backward(epoch, error_gradient);
            i += 1;
        }
        error /= i as f64;
        error
    }

    pub fn train(
        &mut self,
        epoch: usize,
        x_train: &Vec<Vec<f64>>,
        y_train: &Vec<Vec<f64>>,
        loss: &Loss,
        batch_size: usize,
    ) -> f64 {
        self._train(
            epoch,
            x_train
                .into_iter()
                .map(|col| DVector::<f64>::from_iterator(self.i, col.clone().into_iter()))
                .collect(),
            y_train
                .into_iter()
                .map(|col| DVector::<f64>::from_iterator(self.j, col.clone().into_iter()))
                .collect(),
            loss,
            batch_size,
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

    fn backward(&mut self, epoch: usize, error_gradient: DMatrix<f64>) -> DMatrix<f64> {
        let mut error_gradient = error_gradient;
        for layer in self.iter_mut().rev() {
            error_gradient = layer.backward(epoch, error_gradient);
        }
        error_gradient
    }
}
