use crate::{
    linalg::Scalar,
    vision::{conv_layer::full_conv_layer::FullConvLayer, image_activation::ConvActivation},
};
use serde::{Deserialize, Serialize};

use super::conv_layer_spec::ConvLayerSpec;

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct FullConvLayerSpec {
    pub activation: ConvActivation,
    pub dropout: Option<Scalar>,
    pub conv: ConvLayerSpec,
}

impl FullConvLayerSpec {
    pub fn out_size(&self, image_size: (usize, usize)) -> (usize, (usize, usize)) {
        self.conv.out_size(image_size)
    }

    pub fn to_conv_layer(self, nchans: usize) -> FullConvLayer {
        FullConvLayer::new(
            Box::new(self.conv.to_conv_layer(nchans)),
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
            conv: ConvLayerSpec::default(),
        }
    }
}

pub enum FullConvLayerOptions {
    Activation(ConvActivation),
    Dropout(Option<Scalar>),
    Conv(ConvLayerSpec),
}
