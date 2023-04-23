use crate::{
    layer::{full_layer::FullLayer, Layer},
    linalg::{Matrix, MatrixTrait},
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

    /// `input` has shape `(i,)` where `i` is the number of inputs.
    pub fn predict(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.layers.iter_mut().for_each(|l| l.disable_dropout());

        self.layers
            .forward(Matrix::from_column_vector(input))
            .get_column(0)
    }

    /// `input` has shape `(i,)` where `i` is the number of inputs.
    ///
    /// `y` has shape `(j,)` where `j` is the number of outputs.
    pub fn predict_evaluate(
        &mut self,
        input: Vec<f64>,
        y: Vec<f64>,
        loss: &Loss,
    ) -> (Vec<f64>, f64) {
        let preds: Vec<_> = self.predict(&input);
        let loss = loss.loss_vec(&vec![y], &vec![preds.clone()]);

        (preds, loss)
    }

    /// `inputs` has shape `(n, i)` where `n` is the number of samples and `i` is the number of inputs.
    ///
    /// `ys` has shape `(n, j)` where `n` is the number of samples and `j` is the number of outputs.
    ///
    /// Returns a tuple of:
    /// - `preds` which has shape `(n, j)` where `n` is the number of samples and `j` is the number of outputs.
    /// - `avg_loss` which is the average loss over all samples.
    /// - `std_loss` which is the standard deviation of the loss over all samples.
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

    /// `x_train` has shape `(i, n)` where `n` is the number of samples and `i` is the number of inputs.
    ///
    /// `y_train` has shape `(j, n)` where `n` is the number of samples and `j` is the number of outputs.
    ///
    /// Returns the average loss over all samples.
    pub fn train(
        &mut self,
        epoch: usize,
        x_train: &Vec<Vec<f64>>,
        y_train: &Vec<Vec<f64>>,
        loss: &Loss,
        batch_size: usize,
    ) -> f64 {
        self.layers.iter_mut().for_each(|l| l.enable_dropout());

        let mut error = 0.;
        let mut i = 0;
        let x_train_batches: Vec<_> = x_train.chunks(batch_size).map(|c| c.to_vec()).collect();
        let y_train_batches: Vec<_> = y_train.chunks(batch_size).map(|c| c.to_vec()).collect();

        for (input_batch, y_true_batch) in
            x_train_batches.into_iter().zip(y_train_batches.into_iter())
        {
            let input_batch_matrix = Matrix::from_column_leading_matrix(&input_batch);
            let pred = self.layers.forward(input_batch_matrix);
            let y_true_batch_matrix = Matrix::from_column_leading_matrix(&y_true_batch);
            let e = loss.loss(&y_true_batch_matrix, &pred);
            error += e;

            let error_gradient = loss.loss_prime(&y_true_batch_matrix, &pred);
            self.layers.backward(epoch, error_gradient);
            i += 1;
        }
        error /= i as f64;
        error
    }
}

impl<L: Layer> Layer for Vec<L> {
    fn forward(&mut self, input: Matrix) -> Matrix {
        let mut output = input;
        for layer in self.iter_mut() {
            output = layer.forward(output);
        }
        output
    }

    fn backward(&mut self, epoch: usize, error_gradient: Matrix) -> Matrix {
        let mut error_gradient = error_gradient;
        for layer in self.iter_mut().rev() {
            error_gradient = layer.backward(epoch, error_gradient);
        }
        error_gradient
    }
}
