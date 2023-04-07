use super::ActivationLayer;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + libm::exp(-x))
}

fn sigmoid_prime(x: f64) -> f64 {
    let sigofx = sigmoid(x);
    sigofx * (1. - sigofx)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(sigmoid), |m| m.map(sigmoid_prime))
}
