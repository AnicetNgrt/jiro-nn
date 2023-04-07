use super::ActivationLayer;

fn hbt(x: f64) -> f64 {
    let exp = libm::exp(x);
    let exp_neg = libm::exp(-x);
    (exp - exp_neg) / (exp + exp_neg)
}

fn hbt_prime(x: f64) -> f64 {
    1. - libm::pow(hbt(x), 2.)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(|m| m.map(hbt), |m| m.map(hbt_prime))
}
