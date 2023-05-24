use crate::vision::{conv_initializers::ConvInitializers, image_activation::ConvActivation, conv_optimizer::{ConvOptimizers, conv_sgd, conv_momentum, conv_adam}};

use super::network_model::ConvNetworkModel;

pub struct FullDenseConvLayerModel {
    pub kernels_count: usize,
    pub kernels_size: usize,
    pub activation: ConvActivation,
    pub biases_initializer: ConvInitializers,
    pub kernels_initializer: ConvInitializers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_optimizer: ConvOptimizers,
    parent: ConvNetworkModel
}

impl FullDenseConvLayerModel {
    pub fn new(parent: ConvNetworkModel, kernels_count: usize, kernels_size: usize) -> Self {
        Self {
            kernels_count,
            kernels_size,
            activation: ConvActivation::ConvReLU,
            biases_initializer: ConvInitializers::Zeros,
            kernels_initializer: ConvInitializers::GlorotUniform,
            biases_optimizer: conv_sgd(),
            kernels_optimizer: conv_sgd(),
            parent,
        }
    }

    pub fn end(self) -> ConvNetworkModel {
        self.parent.accept_full_dense(self)
    }

    pub fn activation(self, activation: ConvActivation) -> Self {
        Self {
            activation,
            ..self
        }
    }

    pub fn relu(self) -> Self {
        Self {
            activation: ConvActivation::ConvReLU,
            ..self
        }
    }

    pub fn sigmoid(self) -> Self {
        Self {
            activation: ConvActivation::ConvSigmoid,
            ..self
        }
    }

    pub fn tanh(self) -> Self {
        Self {
            activation: ConvActivation::ConvTanh,
            ..self
        }
    }
    
    pub fn linear(self) -> Self {
        Self {
            activation: ConvActivation::ConvLinear,
            ..self
        }
    }

    pub fn initialize_zeros(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::Zeros,
            kernels_initializer: ConvInitializers::Zeros,
            ..self
        }
    }

    pub fn initialize_uniform(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::Uniform,
            kernels_initializer: ConvInitializers::Uniform,
            ..self
        }
    }

    pub fn initialize_uniform_signed(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::UniformSigned,
            kernels_initializer: ConvInitializers::UniformSigned,
            ..self
        }
    }

    pub fn initialize_glorot_uniform(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::GlorotUniform,
            kernels_initializer: ConvInitializers::GlorotUniform,
            ..self
        }
    }

    pub fn biases_initialize_zeros(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::Zeros,
            ..self
        }
    }

    pub fn biases_initialize_uniform(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::Uniform,
            ..self
        }
    }

    pub fn biases_initialize_uniform_signed(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::UniformSigned,
            ..self
        }
    }

    pub fn biases_initialize_glorot_uniform(self) -> Self {
        Self {
            biases_initializer: ConvInitializers::GlorotUniform,
            ..self
        }
    }

    pub fn kernels_initialize_zeros(self) -> Self {
        Self {
            kernels_initializer: ConvInitializers::Zeros,
            ..self
        }
    }

    pub fn kernels_initialize_uniform(self) -> Self {
        Self {
            kernels_initializer: ConvInitializers::Uniform,
            ..self
        }
    }

    pub fn kernels_initialize_uniform_signed(self) -> Self {
        Self {
            kernels_initializer: ConvInitializers::UniformSigned,
            ..self
        }
    }

    pub fn kernels_initialize_glorot_uniform(self) -> Self {
        Self {
            kernels_initializer: ConvInitializers::GlorotUniform,
            ..self
        }
    }

    pub fn sgd(self) -> Self {
        Self {
            biases_optimizer: conv_sgd(),
            kernels_optimizer: conv_sgd(),
            ..self
        }
    }

    pub fn momentum(self) -> Self {
        Self {
            biases_optimizer: conv_momentum(),
            kernels_optimizer: conv_momentum(),
            ..self
        }
    }

    pub fn adam(self) -> Self {
        Self {
            biases_optimizer: conv_adam(),
            kernels_optimizer: conv_adam(),
            ..self
        }
    }

    pub fn biases_optimizer_sgd(self) -> Self {
        Self {
            biases_optimizer: conv_sgd(),
            ..self
        }
    }

    pub fn biases_optimizer_momentum(self) -> Self {
        Self {
            biases_optimizer: conv_momentum(),
            ..self
        }
    }

    pub fn biases_optimizer_adam(self) -> Self {
        Self {
            biases_optimizer: conv_adam(),
            ..self
        }
    }

    pub fn kernels_optimizer_sgd(self) -> Self {
        Self {
            kernels_optimizer: conv_sgd(),
            ..self
        }
    }

    pub fn kernels_optimizer_momentum(self) -> Self {
        Self {
            kernels_optimizer: conv_momentum(),
            ..self
        }
    }

    pub fn kernels_optimizer_adam(self) -> Self {
        Self {
            kernels_optimizer: conv_adam(),
            ..self
        }
    }

    pub fn optimizer(self, optimizer: ConvOptimizers) -> Self {
        Self {
            biases_optimizer: optimizer.clone(),
            kernels_optimizer: optimizer,
            ..self
        }
    }

    pub fn initializer(self, initializer: ConvInitializers) -> Self {
        Self {
            biases_initializer: initializer.clone(),
            kernels_initializer: initializer,
            ..self
        }
    }

    pub fn biases_optimizer(self, optimizer: ConvOptimizers) -> Self {
        Self {
            biases_optimizer: optimizer,
            ..self
        }
    }

    pub fn kernels_optimizer(self, optimizer: ConvOptimizers) -> Self {
        Self {
            kernels_optimizer: optimizer,
            ..self
        }
    }

    pub fn biases_initializer(self, initializer: ConvInitializers) -> Self {
        Self {
            biases_initializer: initializer,
            ..self
        }
    }

    pub fn kernels_initializer(self, initializer: ConvInitializers) -> Self {
        Self {
            kernels_initializer: initializer,
            ..self
        }
    }
}