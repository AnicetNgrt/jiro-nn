use serde::{Serialize, Deserialize};

use crate::vision::{conv_initializers::ConvInitializers, image_activation::ConvActivation, conv_optimizer::{ConvOptimizers, conv_sgd, conv_momentum, conv_adam}, conv_network::ConvNetworkLayer, conv_layer::{direct_conv_layer::DirectConvLayer, full_conv_layer::FullConvLayer}};

use super::conv_network_model::ConvNetworkModelBuilder;


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FullDirectConvLayerModel {
    pub kernels_size: usize,
    pub activation: ConvActivation,
    pub biases_initializer: ConvInitializers,
    pub kernels_initializer: ConvInitializers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_optimizer: ConvOptimizers,
    pub dropout: Option<f32>
}

impl FullDirectConvLayerModel {
    pub fn to_layer(self, in_img_dims: usize, in_channels: usize) -> (usize, usize, Box<dyn ConvNetworkLayer>) {
        let inner_layer = DirectConvLayer::new(
            self.kernels_size,
            self.kernels_size,
            in_channels,
            self.kernels_initializer,
            self.biases_initializer,
            self.kernels_optimizer,
            self.biases_optimizer
        );
        
        let (out_img_dims, _, out_channels) = DirectConvLayer::out_img_dims_and_channels(
            self.kernels_size,
            self.kernels_size,
            in_img_dims,
            in_img_dims,
            in_channels,
        );
        
        let layer = FullConvLayer::new(
            Box::new(inner_layer),
            self.activation.to_layer(),
            self.dropout
        );

        (out_img_dims, out_channels, Box::new(layer))
    }
}

pub struct FullDirectConvLayerModelBuilder {
    pub model: FullDirectConvLayerModel,
    parent: ConvNetworkModelBuilder
}

impl FullDirectConvLayerModelBuilder {
    pub fn new(parent: ConvNetworkModelBuilder, kernels_size: usize) -> Self {
        Self {
            model: FullDirectConvLayerModel {
                kernels_size,
                activation: ConvActivation::ConvReLU,
                biases_initializer: ConvInitializers::Zeros,
                kernels_initializer: ConvInitializers::GlorotUniform,
                biases_optimizer: conv_sgd(),
                kernels_optimizer: conv_sgd(),
                dropout: None
            },
            parent,
        }
    }

    pub fn end(self) -> ConvNetworkModelBuilder {
        self.parent.accept_full_direct(self.model)
    }

    pub fn dropout(self, dropped_rate: f32) -> Self {
        Self {
            model: FullDirectConvLayerModel {
                dropout: Some(dropped_rate),
                ..self.model
            },
            ..self
        }
    }

    pub fn activation(self, activation: ConvActivation) -> Self {
        Self {
            model: FullDirectConvLayerModel {
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
            model: FullDirectConvLayerModel {
                biases_optimizer: optimizer,
                ..self.model
            },
            ..self
        }
    }

    pub fn kernels_optimizer(self, optimizer: ConvOptimizers) -> Self {
        Self {
            model: FullDirectConvLayerModel {
                kernels_optimizer: optimizer,
                ..self.model
            },
            ..self
        }
    }

    pub fn biases_init(self, initializer: ConvInitializers) -> Self {
        Self {
            model: FullDirectConvLayerModel {
                biases_initializer: initializer,
                ..self.model
            },
            ..self
        }
    }

    pub fn kernels_init(self, initializer: ConvInitializers) -> Self {
        Self {
            model: FullDirectConvLayerModel {
                kernels_initializer: initializer,
                ..self.model
            },
            ..self
        }
    }
}