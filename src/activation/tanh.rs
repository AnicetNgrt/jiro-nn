use super::ActivationLayer;

pub fn hyperbolic_tangent(x: f64) -> f64 {
    let exp = libm::exp(x);
    let exp_neg = libm::exp(-x);
    (exp - exp_neg) / (exp + exp_neg)
}

pub fn hyperbolic_tangent_prime(x: f64) -> f64 {
    1. - libm::pow(hyperbolic_tangent(x), 2.)
}

pub fn hyperbolic_tangent_vec(m: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    m.map(hyperbolic_tangent)
}

pub fn hyperbolic_tangent_prime_vec(m: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    m.map(hyperbolic_tangent_prime)
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(hyperbolic_tangent_vec, hyperbolic_tangent_prime_vec)
}
