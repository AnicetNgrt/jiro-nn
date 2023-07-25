use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    model::{impl_model_no_params, Model},
    op_graphs::op_node::OpNodeTrait,
    Data,
};

pub trait MeanableData<'g>: Data<'g> {
    fn mean(&self) -> Scalar;
}

pub struct Loss<'g, D: MeanableData<'g>> {
    pub loss: fn(&D, &D) -> D,
    pub grad: fn(D, &D) -> D,
    pub input: Option<D>,
    _phantom: std::marker::PhantomData<&'g D>,
}

impl<'g, D: MeanableData<'g>> Model for Loss<'g, D> {
    impl_model_no_params!();
}

impl<'g, D: MeanableData<'g>> Loss<'g, D> {
    pub fn new(loss: fn(&D, &D) -> D, grad: fn(D, &D) -> D) -> Self {
        Self {
            loss,
            grad,
            input: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn loss(&self, input: &'g D, reference: &'g D) -> Scalar {
        (self.loss)(input, reference).mean()
    }
}

impl<'g, D: MeanableData<'g>> OpNodeTrait<'g, D, D, D, D> for Loss<'g, D> {
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

    fn revert_reference(&mut self, reference: D) -> D {
        reference
    }
}

impl<'g> MeanableData<'g> for Scalar {
    fn mean(&self) -> Scalar {
        self.clone()
    }
}

impl<'g> MeanableData<'g> for Matrix {
    fn mean(&self) -> Scalar {
        MatrixTrait::mean(self)
    }
}

impl<'g, T: MeanableData<'g>> MeanableData<'g> for Vec<T>
where
    Vec<T>: Data<'g>,
{
    fn mean(&self) -> Scalar {
        self.iter().map(|x| x.mean()).sum::<Scalar>() / self.len() as Scalar
    }
}
