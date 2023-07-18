use crate::linalg::Scalar;

use super::{model::{Model, impl_model_no_params}, Data, LearnableOp};

pub struct Activation<D: Data> {
    f: fn(&D) -> D,
    fp: fn(&D, &D) -> D,
    data_in: Option<D>,
}

impl<D: Data> Activation<D> {
    pub fn new(
        activation: fn(&D) -> D,
        activation_prime: fn(&D, &D) -> D,
    ) -> Self {
        Self {
            f: activation,
            fp: activation_prime,
            data_in: None,
        }
    }
}

impl<D: Data> Model for Activation<D> {
    impl_model_no_params!();
}

impl<D: Data> LearnableOp<D> for Activation<D> {
    fn forward_inference(&mut self, input: D) -> D {
        (self.f)(&input)
    }

    fn forward(&mut self, input: D) -> D {
        (self.f)(&input)
    }

    fn backward(&mut self, incoming_grad: D) -> D {
        let data_in = self.data_in.as_ref().expect("No input: You can't run the backward pass on an Activation layer without running the forward pass first");
        (self.fp)(&data_in, &incoming_grad)
    }
}
