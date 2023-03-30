use nalgebra::SVector;

pub fn mse<const J: usize>(y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> f64 {
    nalgebra_glm::exp2(&(y_true-y_pred)).mean()
}

pub fn mse_prime<const J: usize>(y_true: SVector<f64, J>, y_pred: SVector<f64, J>) -> SVector<f64, J> {
    ((y_pred-y_true) * 2.) / J as f64
}