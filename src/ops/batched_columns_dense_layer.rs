use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    impl_model_op_for_learnable_op,
    matrix_learnable_adam::MatrixLearnableAdamBuilder,
    matrix_learnable_momentum::MatrixLearnableMomentumBuilder,
    matrix_learnable_sgd::MatrixLearnableSGDBuilder,
    model::{impl_model_from_model_fields, Model},
    model_op_builder::{CombinatoryOpBuilder, OpBuild, OpBuilder},
    optimizer::{Optimizer, OptimizerBuilder},
    Data, LearnableOp, ModelOp,
};

pub struct BatchedColumnsDenseLayer {
    weights_optimizer: Box<dyn Optimizer<Matrix>>,
    biases_optimizer: Box<dyn Optimizer<Matrix>>,
    input: Option<Matrix>,
}

impl BatchedColumnsDenseLayer {
    pub fn new(
        weights_optimizer: Box<dyn Optimizer<Matrix>>,
        biases_optimizer: Box<dyn Optimizer<Matrix>>,
    ) -> Self {
        Self {
            weights_optimizer,
            biases_optimizer,
            input: None,
        }
    }
}

impl LearnableOp<Matrix> for BatchedColumnsDenseLayer {
    fn forward_inference(&mut self, input: Matrix) -> Matrix {
        let weights = self.weights_optimizer.get_param();
        let biases = self.biases_optimizer.get_param();

        let res = weights
            .dot(&input)
            .component_add(&biases.dot(&Matrix::constant(1, input.dim().1, 1.0)));

        res
    }

    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = Some(input.clone());
        let res = self.forward_inference(input);
        res
    }

    fn backward(&mut self, incoming_grad: Matrix) -> Matrix {
        let input = self.input.as_ref().unwrap();

        let weights_gradient = &incoming_grad.dot(&input.transpose());

        let biases_gradient = incoming_grad.columns_sum();

        let weights = self.weights_optimizer.get_param();
        let input_gradient = weights.transpose().dot(&incoming_grad);

        self.weights_optimizer.step(weights_gradient);
        self.biases_optimizer.step(&biases_gradient);

        input_gradient
    }
}

impl Model for BatchedColumnsDenseLayer {
    impl_model_from_model_fields!(weights_optimizer, biases_optimizer);
}

impl<DataRef: Data> ModelOp<Matrix, Matrix, DataRef, DataRef> for BatchedColumnsDenseLayer {
    impl_model_op_for_learnable_op!(Matrix, DataRef);
}

pub struct BatchedColumnsDenseLayerBuilder<'a, Parent: 'a> {
    output_neurons: usize,
    weights_optimizer: Box<dyn OptimizerBuilder<Matrix> + 'a>,
    biases_optimizer: Box<dyn OptimizerBuilder<Matrix> + 'a>,
    parent_acceptor:
        Option<Box<dyn FnOnce(BatchedColumnsDenseLayerBuilder<'a, Parent>) -> Parent + 'a>>,
}

impl<'a, Parent: 'a> BatchedColumnsDenseLayerBuilder<'a, Parent> {
    pub fn new(
        output_neurons: usize,
        parent_acceptor: Option<
            Box<dyn FnOnce(BatchedColumnsDenseLayerBuilder<'a, Parent>) -> Parent + 'a>,
        >,
    ) -> Self {
        Self {
            output_neurons,
            weights_optimizer: Box::new(MatrixLearnableAdamBuilder::<Self>::new(None)),
            biases_optimizer: Box::new(MatrixLearnableAdamBuilder::<Self>::new(None)),
            parent_acceptor,
        }
    }

    pub fn with_adam_optimized_weights(mut self) -> MatrixLearnableAdamBuilder<'a, Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_adam_optimized_biases(mut self) -> MatrixLearnableAdamBuilder<'a, Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_adam_optimized(mut self) -> MatrixLearnableAdamBuilder<'a, Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_momentum_optimized_weights(mut self) -> MatrixLearnableMomentumBuilder<'a, Self> {
        MatrixLearnableMomentumBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_momentum_optimized_biases(mut self) -> MatrixLearnableMomentumBuilder<'a, Self> {
        MatrixLearnableMomentumBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_momentum_optimized(mut self) -> MatrixLearnableMomentumBuilder<'a, Self> {
        MatrixLearnableMomentumBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_sgd_optimized_weights(mut self) -> MatrixLearnableSGDBuilder<'a, Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_sgd_optimized_biases(mut self) -> MatrixLearnableSGDBuilder<'a, Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_sgd_optimized(mut self) -> MatrixLearnableSGDBuilder<'a, Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn end(mut self) -> Parent {
        let acceptor = self
            .parent_acceptor
            .take()
            .expect("Can't .end() if there is no parent. .build() instead.");
        (acceptor)(self)
    }
}

impl<'a, Parent: 'a, DataRef: Data> OpBuilder<Matrix, Matrix, DataRef, DataRef>
    for BatchedColumnsDenseLayerBuilder<'a, Parent>
{
    fn build(
        &mut self,
        sample_data: Matrix,
        sample_ref: DataRef,
    ) -> (
        Box<dyn ModelOp<Matrix, Matrix, DataRef, DataRef>>,
        (Matrix, DataRef),
    ) {
        let input_dims = sample_data.dim();
        let input_neurons = input_dims.0;
        let layer = BatchedColumnsDenseLayer::new(
            self.weights_optimizer
                .build(Matrix::zeros(self.output_neurons, input_neurons)),
            self.biases_optimizer
                .build(Matrix::zeros(self.output_neurons, 1)),
        );

        (Box::new(layer), (sample_data, sample_ref))
    }
}

impl<'a, DataIn: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuild<'a, DataIn, Matrix, DataRefIn, DataRefOut>
{
    pub fn dense(
        self,
        output_neurons: usize,
    ) -> BatchedColumnsDenseLayerBuilder<'a, OpBuild<'a, DataIn, Matrix, DataRefIn, DataRefOut>>
    {
        BatchedColumnsDenseLayerBuilder::new(
            output_neurons,
            Some(Box::new(move |builder| {
                let builder = self.push_and_pack(builder);
                builder
            })),
        )
    }
}
