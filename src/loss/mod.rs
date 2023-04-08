use nalgebra::{DMatrix};

pub mod mse;

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
}
