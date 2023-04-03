use super::ActivationLayer;

fn relu(x: f64) -> f64 {
    x.max(0.)
}

fn relu_prime(x: f64) -> f64 {
    if x > 0. {
        1.
    } else {
        0.
    }
}

pub fn new<const I: usize>() -> ActivationLayer<I> {
    ActivationLayer::new(|m| m.map(relu), |m| m.map(relu_prime))
}
