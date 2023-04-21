use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum Initializers {
    Zeros,
    Uniform,
    UniformSigned,
    GlorotUniform,
}

impl Initializers {
    pub fn gen_matrix(&self, i: usize, j: usize) -> DMatrix<f64> {
        match self {
            Initializers::Zeros => DMatrix::<f64>::zeros(j, i),
            Initializers::Uniform => DMatrix::<f64>::new_random(j, i),
            Initializers::UniformSigned => (DMatrix::<f64>::new_random(j, i) * 2.).add_scalar(1.),
            Initializers::GlorotUniform => {
                let limit = (6. / (i + j) as f64).sqrt();
                let range = limit * 2.;
                (DMatrix::<f64>::new_random(j, i) * range).add_scalar(-limit)
            }
        }
    }

    pub fn gen_vector(&self, i: usize) -> DVector<f64> {
        match self {
            Initializers::Zeros => DVector::<f64>::zeros(i),
            Initializers::Uniform => DVector::<f64>::new_random(i),
            Initializers::UniformSigned => (DVector::<f64>::new_random(i) * 2.).add_scalar(1.),
            Initializers::GlorotUniform => {
                // not specified on vectors in the original paper
                // but taken from keras' implementation
                let limit = (6. / (i) as f64).sqrt();
                let range = limit * 2.;
                (DVector::<f64>::new_random(i) * range).add_scalar(-limit)
            }
        }
    }
}