use super::ActivationLayer;
use crate::linalg::{Matrix, MatrixTrait};

// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#:~:text=The%20softmax%20layer%20and%20its%20derivative&text=The%20weight%20matrix%20W%20is,of%20the%20T%20output%20classes.
fn stablesoftmax(m: &Matrix) -> Matrix {
    // let shiftx = m.scalar_sub(m.max());
    // let exps = shiftx.exp();
    // let result = exps.sum();
    todo!()
}

fn stablesoftmax_prime(m: &Matrix) -> Matrix {
    todo!()
}

pub fn new() -> ActivationLayer {
    ActivationLayer::new(stablesoftmax, stablesoftmax_prime)
}
