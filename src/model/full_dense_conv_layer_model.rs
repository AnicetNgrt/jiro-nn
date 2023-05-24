use serde::{Serialize, Deserialize};

use crate::vision::{conv_initializers::ConvInitializers, image_activation::ConvActivation, conv_optimizer::{ConvOptimizers, conv_sgd, conv_momentum, conv_adam}};

use super::conv_network_model::ConvNetworkModelBuilder;

pub struct FullDenseConvLayerModelBuilder {
    pub model: FullDenseConvLayerModel,
    parent: ConvNetworkModelBuilder
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FullDenseConvLayerModel {
    pub kernels_count: usize,
    pub kernels_size: usize,
    pub activation: ConvActivation,
    pub biases_initializer: ConvInitializers,
    pub kernels_initializer: ConvInitializers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_optimizer: ConvOptimizers
}

impl FullDenseConvLayerModelBuilder {
    pub fn new(parent: ConvNetworkModelBuilder, kernels_count: usize, kernels_size: usize) -> Self {
        Self {
            model: FullDenseConvLayerModel {
                kernels_count,
                kernels_size,
                activation: ConvActivation::ConvReLU,
                biases_initializer: ConvInitializers::Zeros,
                kernels_initializer: ConvInitializers::GlorotUniform,
                biases_optimizer: conv_sgd(),
                kernels_optimizer: conv_sgd()
            },
            parent,
        }
    }

    pub fn end(self) -> ConvNetworkModelBuilder {
        self.parent.accept_full_dense(self.model)
    }

    pub fn activation(self, activation: ConvActivation) -> Self {
        Self {
            model: FullDenseConvLayerModel {
                activation,
                ..self.model
            },
            ..self
        }
    }

    pub fn relu(self) -> Self {
        self.activation(ConvActivation::ConvReLU)
    }

    pub fn sigmoid(self) -> Self {
        self.activation(ConvActivation::ConvSigmoid)
    }

    pub fn tanh(self) -> Self {
        self.activation(ConvActivation::ConvTanh)
    }
    
    pub fn linear(self) -> Self {
        self.activation(ConvActivation::ConvLinear)
    }

    pub fn init_zeros(self) -> Self {
        self.init(ConvInitializers::Zeros)
    }

    pub fn init_uniform(self) -> Self {
        self.init(ConvInitializers::Uniform)
    }

    pub fn init_uniform_signed(self) -> Self {
        self.init(ConvInitializers::UniformSigned)
    }

    pub fn init_glorot_uniform(self) -> Self {
        self.init(ConvInitializers::GlorotUniform)
    }

    pub fn biases_init_zeros(self) -> Self {
        self.biases_init(ConvInitializers::Zeros)
    }

    pub fn biases_init_uniform(self) -> Self {
        self.biases_init(ConvInitializers::Uniform)
    }

    pub fn biases_init_uniform_signed(self) -> Self {
        self.biases_init(ConvInitializers::UniformSigned)
    }

    pub fn biases_init_glorot_uniform(self) -> Self {
        self.biases_init(ConvInitializers::GlorotUniform)
    }

    pub fn kernels_init_zeros(self) -> Self {
        self.kernels_init(ConvInitializers::Zeros)
    }

    pub fn kernels_init_uniform(self) -> Self {
        self.biases_init(ConvInitializers::Uniform)
    }

    pub fn kernels_init_uniform_signed(self) -> Self {
        self.biases_init(ConvInitializers::UniformSigned)
    }

    pub fn kernels_init_glorot_uniform(self) -> Self {
        self.biases_init(ConvInitializers::GlorotUniform)
    }

    pub fn sgd(self) -> Self {
        self.optimizer(conv_sgd())
    }

    pub fn momentum(self) -> Self {
        self.optimizer(conv_momentum())
    }

    pub fn adam(self) -> Self {
        self.optimizer(conv_adam())
    }

    pub fn biases_optimizer_sgd(self) -> Self {
        self.biases_optimizer(conv_sgd())
    }

    pub fn biases_optimizer_momentum(self) -> Self {
        self.biases_optimizer(conv_momentum())
    }

    pub fn biases_optimizer_adam(self) -> Self {
        self.biases_optimizer(conv_adam())
    }

    pub fn kernels_optimizer_sgd(self) -> Self {
        self.kernels_optimizer(conv_sgd())
    }

    pub fn kernels_optimizer_momentum(self) -> Self {
        self.kernels_optimizer(conv_momentum())
    }

    pub fn kernels_optimizer_adam(self) -> Self {
        self.kernels_optimizer(conv_adam())
    }

    pub fn optimizer(self, optimizer: ConvOptimizers) -> Self {
        self.biases_optimizer(optimizer.clone())
            .kernels_optimizer(optimizer)
    }

    pub fn init(self, initializer: ConvInitializers) -> Self {
        self.biases_init(initializer.clone())
            .kernels_init(initializer)
    }

    pub fn biases_optimizer(self, optimizer: ConvOptimizers) -> Self {
        Self {
            model: FullDenseConvLayerModel {
                biases_optimizer: optimizer,
                ..self.model
            },
            ..self
        }
    }

    pub fn kernels_optimizer(self, optimizer: ConvOptimizers) -> Self {
        Self {
            model: FullDenseConvLayerModel {
                kernels_optimizer: optimizer,
                ..self.model
            },
            ..self
        }
    }

    pub fn biases_init(self, initializer: ConvInitializers) -> Self {
        Self {
            model: FullDenseConvLayerModel {
                biases_initializer: initializer,
                ..self.model
            },
            ..self
        }
    }

    pub fn kernels_init(self, initializer: ConvInitializers) -> Self {
        Self {
            model: FullDenseConvLayerModel {
                kernels_initializer: initializer,
                ..self.model
            },
            ..self
        }
    }
}