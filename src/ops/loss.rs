use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    model::{impl_model_no_params, Model},
    Data, ModelOp,
};

pub trait MeanableData<'opgraph>: Data<'opgraph> {
    fn mean(&self) -> Scalar;
}

pub struct Loss<'opgraph, D: MeanableData<'opgraph>> {
    pub loss: fn(&D, &D) -> D,
    pub grad: fn(D, &D) -> D,
    pub input: Option<D>,
    _phantom: std::marker::PhantomData<&'opgraph D>,
}

impl<'opgraph, D: MeanableData<'opgraph>> Model for Loss<'opgraph, D> {
    impl_model_no_params!();
}

impl<'opgraph, D: MeanableData<'opgraph>> Loss<'opgraph, D> {
    pub fn new(loss: fn(&D, &D) -> D, grad: fn(D, &D) -> D) -> Self {
        Self {
            loss,
            grad,
            input: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn loss(&self, input: &'opgraph D, reference: &'opgraph D) -> Scalar {
        (self.loss)(input, reference).mean()
    }
}

impl<'opgraph, D: MeanableData<'opgraph>> ModelOp<'opgraph, D, D, D, D> for Loss<'opgraph, D> {
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

impl<'opgraph> MeanableData<'opgraph> for Scalar {
    fn mean(&self) -> Scalar {
        self.clone()
    }
}

impl<'opgraph> MeanableData<'opgraph> for Matrix {
    fn mean(&self) -> Scalar {
        MatrixTrait::mean(self)
    }
}

impl<'opgraph, T: MeanableData<'opgraph>> MeanableData<'opgraph> for Vec<T>
where
    Vec<T>: Data<'opgraph>,
{
    fn mean(&self) -> Scalar {
        self.iter().map(|x| x.mean()).sum::<Scalar>() / self.len() as Scalar
    }
}
