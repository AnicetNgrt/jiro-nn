use nalgebra::SVector;

pub mod mse;

pub trait Loss<const J: usize> {
    fn loss(y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> f64;
    
    fn loss_prime(
        y_true: SVector<f64, J>,
        y_pred: SVector<f64, J>,
    ) -> SVector<f64, J>;
}