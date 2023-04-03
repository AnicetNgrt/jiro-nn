use nalgebra::SVector;

pub mod mse;

pub type LossFn<const J: usize> = fn(SVector<f64, J>, SVector<f64, J>) -> f64;
pub type LossPrimeFn<const J: usize> = fn(SVector<f64, J>, SVector<f64, J>) -> SVector<f64, J>;

pub struct Loss<const J: usize> {
    loss: LossFn<J>,
    derivative: LossPrimeFn<J>,
}

impl<const J: usize> Loss<J> {
    pub fn new(loss: LossFn<J>, derivative: LossPrimeFn<J>) -> Self {
        Self { loss, derivative }
    }
}

impl<const J: usize> Loss<J> {
    pub fn loss(&self, y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> f64 {
        (self.loss)(y_true, y_pred)
    }

    pub fn loss_prime(&self, y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> SVector<f64, J> {
        (self.derivative)(y_true, y_pred)
    }
}
