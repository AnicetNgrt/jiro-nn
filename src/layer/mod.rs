use crate::{activation::Activation, linalg::{Matrix, Scalar}};

pub mod dense_layer;
pub mod full_layer;

pub enum Layers {
    Dense,
    Activation(Activation),
}

pub trait Layer {
    /// `input` has shape `(i, n)` where `i` is the number of inputs and `n` is the number of samples.
    ///
    /// Returns output which has shape `(j, n)` where `j` is the number of outputs and `n` is the number of samples.
    fn forward(&mut self, input: Matrix) -> Matrix;

    /// `output_gradient` has shape `(j, n)` where `j` is the number of outputs and `n` is the number of samples.
    ///
    /// Returns `input_gradient` which has shape `(i, n)` where `i` is the number of inputs and `n` is the number of samples.
    fn backward(&mut self, epoch: usize, output_gradient: Matrix)
        -> Matrix;
}

pub trait ParameterableLayer {
    fn as_learnable_layer(&self) -> Option<&dyn LearnableLayer>;
    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn LearnableLayer>;
    fn as_dropout_layer(&mut self) -> Option<&mut dyn DropoutLayer>;
}

pub trait DropoutLayer {
    fn enable_dropout(&mut self);
    fn disable_dropout(&mut self);
}

pub trait LearnableLayer {
    fn get_learnable_parameters(&self) -> Vec<Vec<Scalar>>;
    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Scalar>>);
}