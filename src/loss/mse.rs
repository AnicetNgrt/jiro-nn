use nalgebra::SVector;

use crate::loss::Loss;

pub struct MeanSquaredError;

impl<const J: usize> Loss<J> for MeanSquaredError {
    fn loss(y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> f64 {
        (y_true - y_pred).map(|y| y*y).sum() / J as f64
    }

    fn loss_prime(
        y_true: SVector<f64, J>,
        y_pred: SVector<f64, J>,
    ) -> SVector<f64, J> {
        ((y_pred-y_true) * 2.) / J as f64
    }
}
