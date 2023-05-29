use serde::{Serialize, Deserialize};

use crate::{activation::Activation, initializers::Initializers, optimizer::{Optimizers, sgd, momentum, adam}, layer::{dense_layer::DenseLayer, full_layer::FullLayer}, network::NetworkLayer};

use super::network_model::NetworkModelBuilder;


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FullDenseLayerModel {
    pub size: usize,
    pub activation: Activation,
    pub biases_initializer: Initializers,
    pub weights_initializer: Initializers,
    pub biases_optimizer: Optimizers,
    pub weights_optimizer: Optimizers,
    pub dropout: Option<f32>
}

impl FullDenseLayerModel {
    pub fn to_layer(self, in_size: usize) -> (usize, Box<dyn NetworkLayer>) {
        let inner = DenseLayer::new(
            in_size,
            self.size,
            self.biases_optimizer,
            self.weights_optimizer,
            self.weights_initializer,
            self.biases_initializer,
        );

        let layer = FullLayer::new(
            inner,
            self.activation.to_layer(),
            self.dropout
        );

        (self.size, Box::new(layer))
    }
}

pub struct FullDenseLayerModelBuilder {
    pub model: FullDenseLayerModel,
    parent: NetworkModelBuilder
}

impl FullDenseLayerModelBuilder {
    pub fn new(parent: NetworkModelBuilder, size: usize) -> Self {
        Self {
            model: FullDenseLayerModel {
                size,
                activation: Activation::ReLU,
                biases_initializer: Initializers::Zeros,
                weights_initializer: Initializers::GlorotUniform,
                biases_optimizer: sgd(),
                weights_optimizer: sgd(),
                dropout: None
            },
            parent,
        }
    }

    pub fn end(self) -> NetworkModelBuilder {
        self.parent.accept_full_dense(self.model)
    }

    pub fn dropout(self, dropout: f32) -> Self {
        Self {
            model: FullDenseLayerModel {
                dropout: Some(dropout),
                ..self.model
            },
            ..self
        }
    }

    pub fn activation(self, activation: Activation) -> Self {
        Self {
            model: FullDenseLayerModel {
                activation,
                ..self.model
            },
            ..self
        }
    }

    pub fn relu(self) -> Self {
        self.activation(Activation::ReLU)
    }

    pub fn sigmoid(self) -> Self {
        self.activation(Activation::Sigmoid)
    }

    pub fn tanh(self) -> Self {
        self.activation(Activation::Tanh)
    }
    
    pub fn linear(self) -> Self {
        self.activation(Activation::Linear)
    }

    pub fn init_zeros(self) -> Self {
        self.init(Initializers::Zeros)
    }

    pub fn init_uniform(self) -> Self {
        self.init(Initializers::Uniform)
    }

    pub fn init_uniform_signed(self) -> Self {
        self.init(Initializers::UniformSigned)
    }

    pub fn init_glorot_uniform(self) -> Self {
        self.init(Initializers::GlorotUniform)
    }

    pub fn biases_init_zeros(self) -> Self {
        self.biases_init(Initializers::Zeros)
    }

    pub fn biases_init_uniform(self) -> Self {
        self.biases_init(Initializers::Uniform)
    }

    pub fn biases_init_uniform_signed(self) -> Self {
        self.biases_init(Initializers::UniformSigned)
    }

    pub fn biases_init_glorot_uniform(self) -> Self {
        self.biases_init(Initializers::GlorotUniform)
    }

    pub fn weights_init_zeros(self) -> Self {
        self.weights_init(Initializers::Zeros)
    }

    pub fn weights_init_uniform(self) -> Self {
        self.biases_init(Initializers::Uniform)
    }

    pub fn weights_init_uniform_signed(self) -> Self {
        self.biases_init(Initializers::UniformSigned)
    }

    pub fn weights_init_glorot_uniform(self) -> Self {
        self.biases_init(Initializers::GlorotUniform)
    }

    pub fn sgd(self) -> Self {
        self.optimizer(sgd())
    }

    pub fn momentum(self) -> Self {
        self.optimizer(momentum())
    }

    pub fn adam(self) -> Self {
        self.optimizer(adam())
    }

    pub fn biases_optimizer_sgd(self) -> Self {
        self.biases_optimizer(sgd())
    }

    pub fn biases_optimizer_momentum(self) -> Self {
        self.biases_optimizer(momentum())
    }

    pub fn biases_optimizer_adam(self) -> Self {
        self.biases_optimizer(adam())
    }

    pub fn weights_optimizer_sgd(self) -> Self {
        self.weights_optimizer(sgd())
    }

    pub fn weights_optimizer_momentum(self) -> Self {
        self.weights_optimizer(momentum())
    }

    pub fn weights_optimizer_adam(self) -> Self {
        self.weights_optimizer(adam())
    }

    pub fn optimizer(self, optimizer: Optimizers) -> Self {
        self.biases_optimizer(optimizer.clone())
            .weights_optimizer(optimizer)
    }

    pub fn init(self, initializer: Initializers) -> Self {
        self.biases_init(initializer.clone())
            .weights_init(initializer)
    }

    pub fn biases_optimizer(self, optimizer: Optimizers) -> Self {
        Self {
            model: FullDenseLayerModel {
                biases_optimizer: optimizer,
                ..self.model
            },
            ..self
        }
    }

    pub fn weights_optimizer(self, optimizer: Optimizers) -> Self {
        Self {
            model: FullDenseLayerModel {
                weights_optimizer: optimizer,
                ..self.model
            },
            ..self
        }
    }

    pub fn biases_init(self, initializer: Initializers) -> Self {
        Self {
            model: FullDenseLayerModel {
                biases_initializer: initializer,
                ..self.model
            },
            ..self
        }
    }

    pub fn weights_init(self, initializer: Initializers) -> Self {
        Self {
            model: FullDenseLayerModel {
                weights_initializer: initializer,
                ..self.model
            },
            ..self
        }
    }
}