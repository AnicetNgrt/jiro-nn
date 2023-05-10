use crate::{
    linalg::Scalar,
    vision::{conv_layer::{full_conv_layer::FullConvLayer, ConvLayer}, image_activation::ConvActivation},
};
use serde::{Deserialize, Serialize};

use super::{dense_conv_layer_spec::DenseConvLayerSpec, direct_conv_layer_spec::DirectConvLayerSpec};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct FullConvLayerSpec {
    pub activation: ConvActivation,
    pub dropout: Option<Scalar>,
    pub conv: ConvLayerSpecTypes,
}

impl FullConvLayerSpec {
    pub fn out_size(&self, image_size: (usize, usize), in_channels: usize) -> (usize, (usize, usize)) {
        self.conv.out_size(image_size, in_channels)
    }

    pub fn to_conv_layer(self, nchans: usize) -> FullConvLayer {
        FullConvLayer::new(
            self.conv.to_conv_layer(nchans),
            self.activation.to_layer(),
            self.dropout,
        )
    }

    /// The `from_options` method is a constructor function for creating a `LayerSpecTypes` enum from a list of `FullConvLayerOptions`.
    ///
    /// See the `FullConvLayerOptions` enum for more information.
    pub fn from_options(options: &[FullConvLayerOptions]) -> FullConvLayerSpec {
        let mut spec = FullConvLayerSpec::default();
        for option in options {
            match option {
                FullConvLayerOptions::Activation(activation) => {
                    spec.activation = activation.clone()
                }
                FullConvLayerOptions::Dropout(dropout) => spec.dropout = dropout.clone(),
                FullConvLayerOptions::Conv(conv) => spec.conv = conv.clone(),
            }
        }
        spec
    }

    pub fn default() -> FullConvLayerSpec {
        FullConvLayerSpec {
            activation: ConvActivation::Linear,
            dropout: None,
            conv: ConvLayerSpecTypes::Dense(DenseConvLayerSpec::default()),
        }
    }
}

pub enum FullConvLayerOptions {
    Activation(ConvActivation),
    Dropout(Option<Scalar>),
    Conv(ConvLayerSpecTypes),
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum ConvLayerSpecTypes {
    Dense(DenseConvLayerSpec),
    Direct(DirectConvLayerSpec)
}

impl ConvLayerSpecTypes {
    pub fn out_size(&self, image_size: (usize, usize), in_channels: usize) -> (usize, (usize, usize)) {
        match self {
            ConvLayerSpecTypes::Dense(spec) => spec.out_size(image_size),
            ConvLayerSpecTypes::Direct(spec) => spec.out_size(image_size, in_channels),
        }
    }

    pub fn to_conv_layer(self, nchans: usize) -> Box<dyn ConvLayer> {
        match self {
            ConvLayerSpecTypes::Dense(spec) => Box::new(spec.to_conv_layer(nchans)),
            ConvLayerSpecTypes::Direct(spec) => Box::new(spec.to_conv_layer(nchans)),
        }
    }
}