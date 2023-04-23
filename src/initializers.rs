use serde::{Deserialize, Serialize};

use crate::linalg::{Matrix, MatrixTrait};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum Initializers {
    Zeros,
    Uniform,
    UniformSigned,
    GlorotUniform,
}

impl Initializers {
    pub fn gen_matrix(&self, nrow: usize, ncol: usize) -> Matrix {
        match self {
            Initializers::Zeros => Matrix::zeros(nrow, ncol),
            Initializers::Uniform => Matrix::random_uniform(nrow, ncol, 0.0, 1.0),
            Initializers::UniformSigned => Matrix::random_uniform(nrow, ncol, -1.0, 1.0),
            Initializers::GlorotUniform => {
                let limit = (6. / (ncol + nrow) as f64).sqrt();
                Matrix::random_uniform(nrow, ncol, -limit, limit)
            }
        }
    }

    pub fn gen_vector(&self, nrow: usize) -> Matrix {
        match self {
            Initializers::Zeros => Matrix::zeros(nrow, 1),
            Initializers::Uniform => Matrix::random_uniform(nrow, 1, 0.0, 1.0),
            Initializers::UniformSigned => Matrix::random_uniform(nrow, 1, -1.0, 1.0),
            Initializers::GlorotUniform => {
                // not specified on vectors in the original paper
                // but taken from keras' implementation
                let limit = (6. / (nrow) as f64).sqrt();
                Matrix::random_uniform(nrow, 1, -limit, limit)
            }
        }
    }
}