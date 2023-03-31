use nalgebra::SVector;

pub trait Loss<const J: usize> {
    fn loss(&self, y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> f64;
    
    fn loss_prime(
        &self,
        y_true: SVector<f64, J>,
        y_pred: SVector<f64, J>,
    ) -> SVector<f64, J>;
}