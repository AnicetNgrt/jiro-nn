use crate::loss::Loss;

pub fn mse(y_true: Vec<f64>, y_pred: Vec<f64>) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(yt, yp)| yt - yp)
        .map(|y| y * y)
        .sum::<f64>()
        / y_true.len() as f64
}

pub fn new<const J: usize>() -> Loss<J> {
    Loss::new(
        |y_true, y_pred| (y_true - y_pred).map(|y| y * y).sum() / J as f64,
        |y_true, y_pred| ((y_pred - y_true) * 2.) / J as f64,
    )
}
