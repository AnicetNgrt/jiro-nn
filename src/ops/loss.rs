use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    model::{impl_model_no_params, Model},
    Data, ModelOp,
};

pub trait MeanableData: Data {
    fn mean(&self) -> Scalar;
}

pub struct Loss<D: MeanableData> {
    pub loss: fn(&D, &D) -> D,
    pub grad: fn(D, &D) -> D,
    pub input: Option<D>,
}

impl<D: MeanableData> Model for Loss<D> {
    impl_model_no_params!();
}

impl<D: MeanableData> Loss<D> {
    pub fn new(loss: fn(&D, &D) -> D, grad: fn(D, &D) -> D) -> Self {
        Self {
            loss,
            grad,
            input: None,
        }
    }
}

impl<D: MeanableData> ModelOp<D, D, D, D> for Loss<D> {
    fn forward_or_transform_inference(&mut self, input: D) -> D {
        input
    }

    fn forward_or_transform(&mut self, input: D, reference: D) -> (D, D) {
        let loss = (self.loss)(&input, &reference);
        self.input = Some(input);
        (loss, reference)
    }

    fn backward_or_revert(&mut self, _loss: D, reference: D) -> (D, D) {
        let input = self
            .input
            .take()
            .expect("Cannot compute the loss gradient before computing the loss itself");
        
        let grad = (self.grad)(input, &reference);
        (grad, reference)
    }
}

impl MeanableData for Scalar {
    fn mean(&self) -> Scalar {
        self.clone()
    }
}

impl MeanableData for Matrix {
    fn mean(&self) -> Scalar {
        MatrixTrait::mean(self)
    }
}

impl<T: MeanableData> MeanableData for Vec<T>
where
    Vec<T>: Data,
{
    fn mean(&self) -> Scalar {
        self.iter().map(|x| x.mean()).sum::<Scalar>() / self.len() as Scalar
    }
}
