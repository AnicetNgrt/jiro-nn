use nalgebra::{DMatrix};
use serde::{Serialize, Deserialize};

pub mod mse;

#[derive(Serialize, Debug, Deserialize, Clone)]
#[serde(tag = "type", content = "params")]
pub enum Losses {
    MSE,
}

impl Losses {
    pub fn to_loss(&self) -> Loss {
        match self {
            Losses::MSE => mse::new(),
        }
    }
}

pub type LossFn = fn(&DMatrix<f64>, &DMatrix<f64>) -> f64;
pub type LossPrimeFn = fn(&DMatrix<f64>, &DMatrix<f64>) -> DMatrix<f64>;

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
    pub fn loss(&self, y_true: &DMatrix<f64>, y_pred: &DMatrix<f64>) -> f64 {
        (self.loss)(y_true, y_pred)
    }

    pub fn loss_prime(&self, y_true: &DMatrix<f64>, y_pred: &DMatrix<f64>) -> DMatrix<f64> {
        (self.derivative)(y_true, y_pred)
    }

    pub fn loss_vec(&self, y_true: &Vec<Vec<f64>>, y_pred: &Vec<Vec<f64>>) -> f64 {
        let y_true = DMatrix::from_row_slice(y_true.len(), y_true[0].len(), &y_true.concat());
        let y_pred = DMatrix::from_row_slice(y_pred.len(), y_pred[0].len(), &y_pred.concat());
        self.loss(&y_true, &y_pred)
    }
}
