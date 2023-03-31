use nalgebra::SVector;

use crate::loss::Loss;

pub struct MeanSquaredError;

impl<const J: usize> Loss<J> for MeanSquaredError {
    fn loss(&self, y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> f64 {
        nalgebra_glm::exp2(&(y_true - y_pred)).mean()
    }

    fn loss_prime(
        &self,
        y_true: SVector<f64, J>,
        y_pred: SVector<f64, J>,
    ) -> SVector<f64, J> {
        ((y_pred - y_true) * 2.) / J as f64
    }
}
