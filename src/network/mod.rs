use std::fmt::Debug;

use crate::{
    layer::{Layer, ParameterableLayer},
    linalg::{Matrix, MatrixTrait, Scalar},
    loss::Loss, introspect::GI,
};

use self::params::NetworkParams;

pub mod params;

#[derive(Debug)]
pub struct Network {
    // May be one or more layers inside
    // A layer is a layer as long as it implements the Layer trait
    layers: Vec<Box<dyn NetworkLayer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn NetworkLayer>>) -> Self {
        Self { layers }
    }

    pub fn get_params(&self) -> NetworkParams {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            layer.as_learnable_layer().map(|l| {
                params.push(l.get_learnable_parameters());
            });
        }
        NetworkParams(params)
    }

    pub fn load_params(&mut self, params: &NetworkParams) {
        for (layer, params) in self.layers.iter_mut().zip(params.0.iter()) {
            layer.as_learnable_layer_mut().map(|l| {
                l.set_learnable_parameters(params);
            });
        }
    }

    /// `input` has shape `(i,)` where `i` is the number of inputs.
    pub fn predict(&mut self, input: &Vec<Scalar>) -> Vec<Scalar> {
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| {
                l.disable_dropout();
            });
        });

        self.layers
            .forward(Matrix::from_column_vector(input))
            .get_column(0)
    }

    /// `input` has shape `(i,)` where `i` is the number of inputs.
    ///
    /// `y` has shape `(j,)` where `j` is the number of outputs.
    pub fn predict_evaluate(
        &mut self,
        input: Vec<Scalar>,
        y: Vec<Scalar>,
        loss: &Loss,
    ) -> (Vec<Scalar>, Scalar) {
        let preds: Vec<_> = self.predict(&input);
        let loss = loss.loss_vec(&vec![y], &vec![preds.clone()]);

        (preds, loss)
    }

    /// `inputs` has shape `(n, i)` where `n` is the number of samples and `i` is the number of inputs.
    ///
    /// Returns `preds` which has shape `(n, j)` where `n` is the number of samples and `j` is the number of outputs.
    pub fn predict_many(&mut self, inputs: &Vec<Vec<Scalar>>) -> Vec<Vec<Scalar>> {
        let preds: Vec<_> = inputs.into_iter().map(|x| self.predict(x)).collect();

        preds
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
        inputs: &Vec<Vec<Scalar>>,
        ys: &Vec<Vec<Scalar>>,
        loss: &Loss,
    ) -> (Vec<Vec<Scalar>>, Scalar, Scalar) {
        let preds_losses: Vec<_> = inputs
            .into_iter()
            .zip(ys.into_iter())
            .map(|(x, y)| self.predict_evaluate(x.clone(), y.clone(), loss))
            .collect();

        let preds = preds_losses.iter().map(|(p, _)| p.clone()).collect();
        let losses: Vec<_> = preds_losses.iter().map(|(_, l)| *l).collect();
        let avg_loss = losses.iter().sum::<Scalar>() / preds_losses.len() as Scalar;
        let std_loss = losses
            .iter()
            .fold(0., |acc, x| acc + (x - avg_loss).powi(2))
            / preds_losses.len() as Scalar;

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
        x_train: &Vec<Vec<Scalar>>,
        y_train: &Vec<Vec<Scalar>>,
        loss: &Loss,
        batch_size: usize,
    ) -> Scalar {
        GI::start_task("train");
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| l.enable_dropout());
        });

        let mut error = 0.;
        let mut i = 0;
        let x_train_batches: Vec<_> = x_train.chunks(batch_size).map(|c| c.to_vec()).collect();
        let y_train_batches: Vec<_> = y_train.chunks(batch_size).map(|c| c.to_vec()).collect();
        let n_batches = x_train_batches.len();
        
        for (input_batch, y_true_batch) in
            x_train_batches.into_iter().zip(y_train_batches.into_iter())
        {
            GI::start_task(format!("batch[{}/{}]", i, n_batches));
            let input_batch_matrix = Matrix::from_column_leading_matrix(&input_batch);

            let pred = self.layers.forward(input_batch_matrix);
            
            let y_true_batch_matrix = Matrix::from_column_leading_matrix(&y_true_batch);
            let e = loss.loss(&y_true_batch_matrix, &pred);

            error += e;

            let error_gradient = loss.loss_prime(&y_true_batch_matrix, &pred);
            self.layers.backward(epoch, error_gradient);
            i += 1;
            GI::end_task();
        }
        error /= i as Scalar;
        GI::end_task();
        error
    }
}

impl Layer for Vec<Box<dyn NetworkLayer>> {
    fn forward(&mut self, input: Matrix) -> Matrix {
        GI::start_task("net.forw");
        let mut output = input;
        let n_layers = self.len();
        for (i, layer) in self.iter_mut().enumerate() {
            GI::start_task(format!("layer[{}/{}]", i+1, n_layers));
            output = layer.forward(output);
            GI::end_task();
        }
        GI::end_task();
        output
    }

    fn backward(&mut self, epoch: usize, error_gradient: Matrix) -> Matrix {
        GI::start_task("net.back");
        let mut error_gradient = error_gradient;
        for (i, layer) in self.iter_mut().enumerate().rev() {
            GI::start_task(format!("layer[{}]", i+1));
            error_gradient = layer.backward(epoch, error_gradient);
            GI::end_task();
        }
        GI::end_task();
        error_gradient
    }
}

pub trait NetworkLayer: Layer + ParameterableLayer + Debug + Send {}
