use crate::{linalg::{Scalar, Matrix}, vision::image::Image};

pub mod batched_vecs_dense_layer;
pub mod matrix_learnable;
pub mod matrix_learnable_sgd;

pub trait Data: 'static {}

pub trait Op<DataIn: Data, DataOut: Data> {
    fn forward_inference(&self, input: &DataIn) -> DataOut;
    fn forward(&mut self, input: &DataIn) -> DataOut;
    fn backward(&mut self, incoming_grad: &DataOut) -> DataIn;
    fn get_learnable_params_count(&self) -> usize;
    fn load_learnable_params(&mut self, params: Vec<Scalar>);
    fn get_learnable_params(&self) -> Vec<Scalar>;
}

pub struct OpChain<DataIn: Data, DataMid: Data, DataOut: Data> {
    first_op: Box<dyn Op<DataIn, DataMid>>,
    second_op: Box<dyn Op<DataMid, DataOut>>,
}

impl<DataIn: Data, DataMid: Data, DataOut: Data> Op<DataIn, DataOut>
    for OpChain<DataIn, DataMid, DataOut>
{
    fn forward_inference(&self, input: &DataIn) -> DataOut {
        let mid = self.first_op.forward_inference(input);
        self.second_op.forward_inference(&mid)
    }

    fn forward(&mut self, input: &DataIn) -> DataOut {
        let mid = self.first_op.forward(input);
        self.second_op.forward(&mid)
    }

    fn backward(&mut self, incoming_grad: &DataOut) -> DataIn {
        let mid_grad = self.second_op.backward(incoming_grad);
        self.first_op.backward(&mid_grad)
    }

    fn get_learnable_params_count(&self) -> usize {
        self.first_op.get_learnable_params_count() + self.second_op.get_learnable_params_count()
    }

    fn load_learnable_params(&mut self, params: Vec<Scalar>) {
        let first_op_params_count = self.first_op.get_learnable_params_count();
        let first_op_params = params[0..first_op_params_count].to_vec();
        let second_op_params = params[first_op_params_count..].to_vec();
        self.first_op.load_learnable_params(first_op_params);
        self.second_op.load_learnable_params(second_op_params);
    }

    fn get_learnable_params(&self) -> Vec<Scalar> {
        let mut params = self.first_op.get_learnable_params();
        params.extend(self.second_op.get_learnable_params());
        params
    }
}

impl<DataIn: Data, DataMid: Data, DataOut: Data> OpChain<DataIn, DataMid, DataOut> {
    pub fn new(
        first_op: Box<dyn Op<DataIn, DataMid>>,
        second_op: Box<dyn Op<DataMid, DataOut>>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }

    pub fn push<DataOutPushed: Data, OpPushed: Op<DataOut, DataOutPushed> + 'static>(
        self,
        op: OpPushed,
    ) -> OpChain<DataIn, DataOut, DataOutPushed> {
        OpChain::new(Box::new(self), Box::new(op))
    }
}

impl Data for Scalar {}
impl Data for Vec<Scalar> {}
impl Data for Vec<Vec<Scalar>> {}
impl Data for Matrix {}
impl Data for Vec<Matrix> {}
impl Data for Vec<Vec<Matrix>> {}
impl Data for Image {}
impl Data for Vec<Image> {}
impl Data for Vec<Vec<Image>> {}