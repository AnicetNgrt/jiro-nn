use crate::loss::Loss;

pub fn new<const J: usize>() -> Loss<J> {
    Loss::new(
        |y_true, y_pred| (y_true - y_pred).map(|y| y * y).sum() / J as f64,
        |y_true, y_pred| ((y_pred - y_true) * 2.) / J as f64,
    )
}
