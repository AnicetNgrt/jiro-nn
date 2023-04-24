use crate::linalg::MatrixTrait;
use crate::{
    initializers::Initializers,
    layer::Layer,
    learning_rate::default_learning_rate,
    linalg::{Matrix, Scalar},
    optimizer::{sgd::SGD, Optimizers},
};

pub struct DenseLayer {
    // i inputs, j outputs, i x j connections
    input: Option<Matrix>,
    // j x i connection weights
    weights: Matrix,
    // j output biases (single column)
    biases: Matrix,
    weights_optimizer: Optimizers,
    biases_optimizer: Optimizers,
}

impl DenseLayer {
    pub fn new(
        i: usize,
        j: usize,
        weights_optimizer: Optimizers,
        biases_optimizer: Optimizers,
        weights_initializer: Initializers,
        biases_initializer: Initializers,
    ) -> Self {
        // about weights initialization : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        let weights = weights_initializer.gen_matrix(j, i);
        let biases = biases_initializer.gen_vector(j);

        Self {
            weights: weights,
            biases: biases,
            input: None,
            weights_optimizer,
            biases_optimizer,
        }
    }

    pub fn map_weights(&mut self, f: impl Fn(Scalar) -> Scalar + Sync) {
        self.weights = self.weights.map(f);
    }
}

impl Layer for DenseLayer {
    /// `input` has shape `(i, n)` where `i` is the number of inputs and `n` is the number of samples.
    ///
    /// Returns output which has shape `(j, n)` where `j` is the number of outputs and `n` is the number of samples.
    fn forward(&mut self, input: Matrix) -> Matrix {
        // Y = W . X + B
        let mut res = self.weights.dot(&input);

        let biases = self.biases.get_column(0);

        res.map_indexed_mut(|i, _, v| v + biases[i]);

        // Old code for fixing & comparison purposes
        //
        // let res = res.columns_map(|i, col| {
        //     col.iter().map(|x| x + biases[i]).collect()
        // });
        //
        // res.column_iter_mut()
        //     .zip(self.biases.iter())
        //     .for_each(|(mut col, bias)| {
        //         col.add_scalar_mut(*bias);
        //     });

        self.input = Some(input);
        res
    }

    /// `output_gradient` has shape `(j, n)` where `j` is the number of outputs and `n` is the number of samples.
    ///
    /// Returns `input_gradient` which has shape `(i, n)` where `i` is the number of inputs and `n` is the number of samples.
    fn backward(&mut self, epoch: usize, output_gradient: Matrix) -> Matrix {
        let input = self.input.clone().unwrap();

        let weights_gradient = &output_gradient.dot(&input.transpose());

        let biases_gradient = Matrix::from_column_vector(&output_gradient.columns_sum());

        let input_gradient = self.weights.transpose().dot(&output_gradient);

        self.weights =
            self.weights_optimizer
                .update_parameters(epoch, &self.weights, &weights_gradient);
        self.biases =
            self.biases_optimizer
                .update_parameters(epoch, &self.biases, &biases_gradient);

        input_gradient
    }
}

pub fn default_biases_initializer() -> Initializers {
    Initializers::Zeros
}

pub fn default_weights_initializer() -> Initializers {
    Initializers::GlorotUniform
}

pub fn default_biases_optimizer() -> Optimizers {
    Optimizers::SGD(SGD::new(default_learning_rate()))
}

pub fn default_weights_optimizer() -> Optimizers {
    Optimizers::SGD(SGD::new(default_learning_rate()))
}
