use nalgebra::DVector;

pub mod mse;

pub type LossFn = fn(&DVector<f64>, &DVector<f64>) -> f64;
pub type LossPrimeFn = fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>;

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
    pub fn loss(&self, y_true: &DVector<f64>, y_pred: &DVector<f64>) -> f64 {
        (self.loss)(y_true, y_pred)
    }

    pub fn loss_prime(&self, y_true: &DVector<f64>, y_pred: &DVector<f64>) -> DVector<f64> {
        (self.derivative)(y_true, y_pred)
    }
}
