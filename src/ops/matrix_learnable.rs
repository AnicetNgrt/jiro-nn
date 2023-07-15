use crate::linalg::Matrix;

use super::Op;

pub trait MatrixLearnable: Op<Matrix, Matrix> {
    fn get(&self) -> &Matrix;
}