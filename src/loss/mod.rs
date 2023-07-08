use serde::{Deserialize, Serialize};

use crate::linalg::Matrix;
use crate::linalg::MatrixTrait;
use crate::linalg::Scalar;

pub mod mse;
pub mod bce;

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum Losses {
    MSE,
    BCE,
}

impl Losses {
    pub fn to_loss(&self) -> Loss {
        match self {
            Losses::MSE => mse::new(),
            Losses::BCE => bce::new(),
        }
    }
}

pub type LossFn = fn(&Matrix, &Matrix) -> Scalar;
pub type LossPrimeFn = fn(&Matrix, &Matrix) -> Matrix;

pub struct Loss {
    loss: LossFn,
    derivative: LossPrimeFn,
}

impl Loss {
    pub fn new(loss: LossFn, derivative: LossPrimeFn) -> Self {
        Self { loss, derivative }
    }
}

impl Loss {
    pub fn loss(&self, y_true: &Matrix, y_pred: &Matrix) -> Scalar {
        (self.loss)(y_true, y_pred)
    }

    pub fn loss_prime(&self, y_true: &Matrix, y_pred: &Matrix) -> Matrix {
        (self.derivative)(y_true, y_pred)
    }

    pub fn loss_vec(&self, y_true: &Vec<Vec<Scalar>>, y_pred: &Vec<Vec<Scalar>>) -> Scalar {
        let y_true = Matrix::from_row_leading_vec2(&y_true);
        let y_pred = Matrix::from_row_leading_vec2(&y_pred);
        self.loss(&y_true, &y_pred)
    }
}
