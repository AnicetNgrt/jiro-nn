use std::fmt::Debug;

use crate::{
    layer::{Layer, ParameterableLayer},
    linalg::{Matrix, MatrixTrait, Scalar},
    loss::Loss, monitor::TM,
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
    pub fn predict_many(&mut self, inputs: &Vec<Vec<Scalar>>, batch_size: usize) -> Vec<Vec<Scalar>> {
        TM::start("predmany");
        TM::start("init");
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| l.disable_dropout());
        });

        let mut preds = vec![];
        let mut i = 0;
        let x_batches: Vec<_> = inputs.chunks(batch_size).map(|c| c.to_vec()).collect();
        let n_batches = x_batches.len();
        TM::end();

        TM::start("batches");
        for input_batch in x_batches.into_iter()
        {
            TM::start(format!("{}/{}", i, n_batches));
            let input_batch_matrix = Matrix::from_column_leading_vector2(&input_batch);
            let pred = self.layers.forward(input_batch_matrix);
            preds.extend(pred.get_data_col_leading());
            i += 1;
            TM::end();
        }
        TM::end();
        TM::end();

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
        batch_size: usize
    ) -> (Vec<Vec<Scalar>>, Scalar, Scalar) {
        TM::start("predevmany");
        TM::start("init");
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| l.disable_dropout());
        });

        let mut losses = vec![];
        let mut preds = vec![];
        let mut i = 0;
        let x_batches: Vec<_> = inputs.chunks(batch_size).map(|c| c.to_vec()).collect();
        let y_batches: Vec<_> = ys.chunks(batch_size).map(|c| c.to_vec()).collect();
        let n_batches = x_batches.len();
        TM::end();
        
        TM::start("batches");
        for (input_batch, y_true_batch) in
            x_batches.into_iter().zip(y_batches.into_iter())
        {
            TM::start(format!("{}/{}", i, n_batches));
            let input_batch_matrix = Matrix::from_column_leading_vector2(&input_batch);
            let pred = self.layers.forward(input_batch_matrix);
            let y_true_batch_matrix = Matrix::from_column_leading_vector2(&y_true_batch);
            let e = loss.loss(&y_true_batch_matrix, &pred);

            losses.push(e);
            preds.extend(pred.get_data_col_leading());
            i += 1;
            TM::end();
        }
        TM::end();

        TM::start("stats");
        let avg_loss = losses.iter().sum::<Scalar>() / losses.len() as Scalar;
        let std_loss = losses
            .iter()
            .fold(0., |acc, x| acc + (x - avg_loss).powi(2))
            / losses.len() as Scalar;
        TM::end();
        
        TM::end_with_message(format!("avg_loss: {}, std_loss: {}", avg_loss, std_loss));
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
        TM::start("train");
        TM::start("init");
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| l.enable_dropout());
        });

        let mut error = 0.;
        let mut i = 0;
        let x_train_batches: Vec<_> = x_train.chunks(batch_size).map(|c| c.to_vec()).collect();
        let y_train_batches: Vec<_> = y_train.chunks(batch_size).map(|c| c.to_vec()).collect();
        let n_batches = x_train_batches.len();
        TM::end();
        
        TM::start("batches");
        for (input_batch, y_true_batch) in
            x_train_batches.into_iter().zip(y_train_batches.into_iter())
        {
            TM::start(format!("{}/{}", i, n_batches));
            let input_batch_matrix = Matrix::from_column_leading_vector2(&input_batch);

            let pred = self.layers.forward(input_batch_matrix);
            
            let y_true_batch_matrix = Matrix::from_column_leading_vector2(&y_true_batch);
            let e = loss.loss(&y_true_batch_matrix, &pred);

            error += e;

            let error_gradient = loss.loss_prime(&y_true_batch_matrix, &pred);
            self.layers.backward(epoch, error_gradient.clone());
            i += 1;
            TM::end_with_message(format!("error: {:.4} total_error: {:.4}", e, error));
        }
        error /= i as Scalar;
        TM::end();
        TM::end_with_message(format!("avg_error: {:.4}", error));
        error
    }
}

impl Layer for Vec<Box<dyn NetworkLayer>> {
    fn forward(&mut self, input: Matrix) -> Matrix {
        TM::start("net.forw");
        let mut output = input;
        let n_layers = self.len();
        for (i, layer) in self.iter_mut().enumerate() {
            TM::start(format!("layer[{}/{}]", i+1, n_layers));
            output = layer.forward(output);
            TM::end();
        }
        TM::end();
        output
    }

    fn backward(&mut self, epoch: usize, error_gradient: Matrix) -> Matrix {
        TM::start("net.back");
        let mut error_gradient = error_gradient;
        for (i, layer) in self.iter_mut().enumerate().rev() {
            TM::start(format!("layer[{}]", i+1));
            error_gradient = layer.backward(epoch, error_gradient);
            TM::end();
        }
        TM::end();
        error_gradient
    }
}

pub trait NetworkLayer: Layer + ParameterableLayer + Debug + Send {}
