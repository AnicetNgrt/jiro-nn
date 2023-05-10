use serde::{Deserialize, Serialize};

use crate::initializers::Initializers;
use crate::layer::defaults::{
    default_biases_initializer, default_biases_optimizer, default_weights_initializer,
    default_weights_optimizer,
};
use crate::layer::dense_layer::DenseLayer;
use crate::layer::full_layer::FullLayer;
use crate::linalg::Scalar;
use crate::{activation::Activation, optimizer::Optimizers};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct FullLayerSpec {
    #[serde(default)]
    pub out_size: usize,
    pub activation: Activation,
    pub dropout: Option<Scalar>,
    #[serde(default = "default_weights_optimizer")]
    pub weights_optimizer: Optimizers,
    #[serde(default = "default_biases_optimizer")]
    pub biases_optimizer: Optimizers,
    #[serde(default = "default_weights_initializer")]
    pub weights_initializer: Initializers,
    #[serde(default = "default_biases_initializer")]
    pub biases_initializer: Initializers,
}

/// Options to be used in order to specify the properties of a layer.
///
/// **Required options**:
///
/// - `OutSize`: The size of the output layer.
///
/// **Optional options**:
///
/// - `Activation`: The activation function to be used (a variant of the `Activation` enum). If ommited defaults to `Activation::Linear`.
/// - `Dropout`: The dropout rate (an optional float). If ommited defaults to `None`.
/// - `WeightsOptimizer`: The optimizer to be used for updating the weights (a variant of the `Optimizers` enum). If ommited defaults to `optimizer::sgd`.
/// - `BiasesOptimizer`: The optimizer to be used for updating the biases (a variant of the `Optimizers` enum). If ommited defaults to `optimizer::sgd`.
/// - `Optimizer`: The optimizer to be used for updating the weights and biases (a variant of the `Optimizers` enum). If ommited defaults to `optimizer::sgd`.
/// - `WeightsInitializer`: The initializer to be used for initializing the weights (a variant of the `Initializers` enum). If ommited defaults to `initializer::GlorotUniform`.
/// - `BiasesInitializer`: The initializer to be used for initializing the biases (a variant of the `Initializers` enum). If ommited defaults to `initializer::Zeros`.
pub enum FullLayerOptions {
    /// The `OutSize` option specifies the size of the output layer.
    OutSize(usize),
    /// The `Activation` option specifies the activation function to be used (a variant of the `Activation` enum). If ommited defaults to `Activation::Linear`.
    Activation(Activation),
    /// The `Dropout` option specifies the dropout rate (an optional float). If ommited defaults to `None`.
    Dropout(Option<Scalar>),
    /// The `WeightsOptimizer` option specifies the optimizer to be used for updating the weights (a variant of the `Optimizers` enum). If ommited defaults to `optimizer::sgd`.
    WeightsOptimizer(Optimizers),
    /// The `BiasesOptimizer` option specifies the optimizer to be used for updating the biases (a variant of the `Optimizers` enum). If ommited defaults to `optimizer::sgd`.
    BiasesOptimizer(Optimizers),
    /// The `Optimizer` option specifies the optimizer to be used for updating the weights and biases (a variant of the `Optimizers` enum). If ommited defaults to `optimizer::sgd`.
    Optimizer(Optimizers),
    /// The `WeightsInitializer` option specifies the initializer to be used for initializing the weights (a variant of the `Initializers` enum). If ommited defaults to `initializer::GlorotUniform`.
    WeightsInitializer(Initializers),
    /// The `BiasesInitializer` option specifies the initializer to be used for initializing the biases (a variant of the `Initializers` enum). If ommited defaults to `initializer::Zeros`.
    BiasesInitializer(Initializers),
}

impl FullLayerSpec {
    pub fn to_layer(self, in_size: usize) -> FullLayer {
        FullLayer::new(
            DenseLayer::new(
                in_size,
                self.out_size,
                self.weights_optimizer,
                self.biases_optimizer,
                self.weights_initializer,
                self.biases_initializer,
            ),
            self.activation.to_layer(),
            self.dropout,
        )
    }

    /// The `from_options` method is a constructor function for creating a `LayerSpecTypes` enum from a list of `FullLayerOptions`.
    ///
    /// See the `FullLayerOptions` enum for more information.
    pub fn from_options(options: &[FullLayerOptions]) -> FullLayerSpec {
        let mut spec = FullLayerSpec::default();
        for option in options {
            match option {
                FullLayerOptions::OutSize(out_size) => spec.out_size = out_size.clone(),
                FullLayerOptions::Activation(activation) => spec.activation = activation.clone(),
                FullLayerOptions::Dropout(dropout) => spec.dropout = dropout.clone(),
                FullLayerOptions::WeightsOptimizer(weights_optimizer) => {
                    spec.weights_optimizer = weights_optimizer.clone()
                }
                FullLayerOptions::BiasesOptimizer(biases_optimizer) => {
                    spec.biases_optimizer = biases_optimizer.clone()
                }
                FullLayerOptions::WeightsInitializer(weights_initializer) => {
                    spec.weights_initializer = weights_initializer.clone()
                }
                FullLayerOptions::BiasesInitializer(biases_initializer) => {
                    spec.biases_initializer = biases_initializer.clone()
                }
                FullLayerOptions::Optimizer(global_optimizer) => {
                    spec.weights_optimizer = global_optimizer.clone();
                    spec.biases_optimizer = global_optimizer.clone();
                }
            }
        }
        spec
    }

    pub fn default() -> FullLayerSpec {
        FullLayerSpec {
            out_size: 0,
            activation: Activation::Linear,
            dropout: None,
            weights_optimizer: default_weights_optimizer(),
            biases_optimizer: default_biases_optimizer(),
            weights_initializer: default_weights_initializer(),
            biases_initializer: default_biases_initializer(),
        }
    }
}
