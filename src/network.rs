use std::{fs::File, io::Write, path::{PathBuf}};

use crate::{
    layer::{full_layer::FullLayer, Layer},
    linalg::{Matrix, MatrixTrait, Scalar},
    loss::Loss,
};

#[derive(Debug)]
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

    pub fn get_params(&self) -> NetworkParams {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            params.push(layer.get_learnable_parameters());
        }
        NetworkParams(params)
    }

    pub fn load_params(&mut self, params: &NetworkParams) {
        for (layer, params) in self.layers.iter_mut().zip(params.0.iter()) {
            layer.set_learnable_parameters(params);
        }
    }

    /// `input` has shape `(i,)` where `i` is the number of inputs.
    pub fn predict(&mut self, input: &Vec<Scalar>) -> Vec<Scalar> {
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
    pub fn predict_many(
        &mut self,
        inputs: &Vec<Vec<Scalar>>
    ) -> Vec<Vec<Scalar>> {
        let preds: Vec<_> = inputs
            .into_iter()
            .map(|x| self.predict(x))
            .collect();

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
        error /= i as Scalar;
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

pub struct NetworkParams(pub Vec<Vec<Vec<Scalar>>>);

impl NetworkParams {
    pub fn average(networks: &Vec<Self>) -> Self {
        let mut params = Vec::new();
        
        let layer_count = networks[0].0.len();

        for layer_index in 0..layer_count {
            let mut layer_params = Matrix::from_column_leading_matrix(&networks[0].0[layer_index]);

            for network in networks.iter().skip(1) {
                let other_params = Matrix::from_column_leading_matrix(&network.0[layer_index]);
                layer_params =  layer_params.component_add(&other_params).scalar_div(2.0);
            }

            params.push(layer_params.get_data());
        }
    
        NetworkParams(params)
    }

    pub fn to_json<P: Into<PathBuf>>(&self, path: P) {
        let json = serde_json::json!({ "params": self.0 });
        let mut file = File::create(path.into()).unwrap();
        file.write_all(json.to_string().as_bytes()).unwrap();
    }

    pub fn from_json<P: Into<PathBuf>>(path: P) -> Self {
        let file = File::open(path.into()).unwrap();
        let json: serde_json::Value = serde_json::from_reader(file).unwrap();
        let params = json["params"].as_array().unwrap();

        NetworkParams(
            params
                .iter()
                .map(|x| {
                    x.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| {
                            x.as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_f64().unwrap() as Scalar)
                                .collect()
                        })
                        .collect()
                })
                .collect(),
        )
    }
}
