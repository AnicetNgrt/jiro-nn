use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::{
    matrix_learnable_adam::MatrixLearnableAdamBuilder,
    matrix_learnable_momentum::MatrixLearnableMomentumBuilder,
    matrix_learnable_sgd::MatrixLearnableSGDBuilder,
    model::{impl_model_from_model_fields, Model},
    op_graph_builder::{CombinatoryOpBuilder, OpGraphBuilder, OpNodeBuilder},
    op_graphs::op_node::{impl_op_node_for_learnable_op, LearnableOp, OpNodeTrait},
    optimizer::{Optimizer, OptimizerBuilder},
    Data,
};

pub struct BatchedColumnsDenseLayer<'g> {
    weights_optimizer: Box<dyn Optimizer<'g, Matrix> + 'g>,
    biases_optimizer: Box<dyn Optimizer<'g, Matrix> + 'g>,
    input: Option<Matrix>,
}

impl<'g> BatchedColumnsDenseLayer<'g> {
    pub fn new(
        weights_optimizer: Box<dyn Optimizer<'g, Matrix> + 'g>,
        biases_optimizer: Box<dyn Optimizer<'g, Matrix> + 'g>,
    ) -> Self {
        Self {
            weights_optimizer,
            biases_optimizer,
            input: None,
        }
    }
}

impl<'g> LearnableOp<'g, Matrix> for BatchedColumnsDenseLayer<'g> {
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

impl<'g> Model for BatchedColumnsDenseLayer<'g> {
    impl_model_from_model_fields!(weights_optimizer, biases_optimizer);
}

impl<'g, DataRef: Data<'g>> OpNodeTrait<'g, Matrix, Matrix, DataRef, DataRef>
    for BatchedColumnsDenseLayer<'g>
{
    impl_op_node_for_learnable_op!(Matrix, DataRef);
}

pub struct BatchedColumnsDenseLayerBuilder<'g, Parent: 'g> {
    output_neurons: usize,
    weights_optimizer: Box<dyn OptimizerBuilder<'g, Matrix> + 'g>,
    biases_optimizer: Box<dyn OptimizerBuilder<'g, Matrix> + 'g>,
    parent_acceptor:
        Option<Box<dyn FnOnce(BatchedColumnsDenseLayerBuilder<'g, Parent>) -> Parent + 'g>>,
}

impl<'g, Parent: 'g> BatchedColumnsDenseLayerBuilder<'g, Parent> {
    pub fn new(
        output_neurons: usize,
        parent_acceptor: Option<
            Box<dyn FnOnce(BatchedColumnsDenseLayerBuilder<'g, Parent>) -> Parent + 'g>,
        >,
    ) -> Self {
        Self {
            output_neurons,
            weights_optimizer: Box::new(MatrixLearnableAdamBuilder::<Self>::new(None)),
            biases_optimizer: Box::new(MatrixLearnableAdamBuilder::<Self>::new(None)),
            parent_acceptor,
        }
    }

    pub fn with_adam_optimized_weights(mut self) -> MatrixLearnableAdamBuilder<'g, Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_adam_optimized_biases(mut self) -> MatrixLearnableAdamBuilder<'g, Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_adam_optimized(mut self) -> MatrixLearnableAdamBuilder<'g, Self> {
        MatrixLearnableAdamBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_momentum_optimized_weights(mut self) -> MatrixLearnableMomentumBuilder<'g, Self> {
        MatrixLearnableMomentumBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_momentum_optimized_biases(mut self) -> MatrixLearnableMomentumBuilder<'g, Self> {
        MatrixLearnableMomentumBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_momentum_optimized(mut self) -> MatrixLearnableMomentumBuilder<'g, Self> {
        MatrixLearnableMomentumBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = builder.clone_box();
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_sgd_optimized_weights(mut self) -> MatrixLearnableSGDBuilder<'g, Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.weights_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn with_sgd_optimized_biases(mut self) -> MatrixLearnableSGDBuilder<'g, Self> {
        MatrixLearnableSGDBuilder::new(Some(Box::new(move |builder| {
            self.biases_optimizer = Box::new(builder);
            self
        })))
    }

    pub fn everything_sgd_optimized(mut self) -> MatrixLearnableSGDBuilder<'g, Self> {
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

impl<'g, Parent: 'g, DataRef: Data<'g>> OpNodeBuilder<'g, Matrix, Matrix, DataRef, DataRef>
    for BatchedColumnsDenseLayerBuilder<'g, Parent>
{
    fn build(
        &mut self,
        meta_data: (usize, usize),
        meta_ref: DataRef::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, Matrix, Matrix, DataRef, DataRef> + 'g>,
        ((usize, usize), DataRef::Meta),
    ) {
        let input_dims = meta_data;
        let input_neurons = input_dims.0;
        let weights_optimizer = self
            .weights_optimizer
            .build(Matrix::zeros(self.output_neurons, input_neurons));
        let biases_optimizer = self
            .biases_optimizer
            .build(Matrix::zeros(self.output_neurons, 1));
        let layer = BatchedColumnsDenseLayer::new(weights_optimizer, biases_optimizer);

        (
            Box::new(layer),
            ((self.output_neurons, input_dims.1), meta_ref),
        )
    }
}

impl<'g, DataIn: Data<'g>, DataRefIn: Data<'g>, DataRefOut: Data<'g>>
    OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut>
{
    pub fn dense(
        self,
        output_neurons: usize,
    ) -> BatchedColumnsDenseLayerBuilder<
        'g,
        OpGraphBuilder<'g, DataIn, Matrix, DataRefIn, DataRefOut>,
    > {
        BatchedColumnsDenseLayerBuilder::new(
            output_neurons,
            Some(Box::new(move |builder| {
                let builder = self.link_and_pack(builder);
                builder
            })),
        )
    }
}
