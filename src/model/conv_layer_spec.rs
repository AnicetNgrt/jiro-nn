use serde::{Deserialize, Serialize};
use crate::{linalg::Scalar, vision::{conv_activation::ConvActivation, conv_optimizer::ConvOptimizers, conv_initializers::ConvInitializers, conv_layer::{full_conv_layer::FullConvLayer, dense_conv_layer::DenseConvLayer}}};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ConvLayerSpec {
    pub kern_size: (usize, usize),
    pub kern_count: usize,
    pub activation: ConvActivation,
    pub dropout: Option<Scalar>,
    pub kernels_optimizer: ConvOptimizers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_initializer: ConvInitializers,
    pub biases_initializer: ConvInitializers,
}

impl ConvLayerSpec {
    pub fn to_conv_layer(self, nchans: usize) -> FullConvLayer {
        FullConvLayer::new(
            DenseConvLayer::new(
                self.kern_size.0,
                self.kern_size.1,
                nchans,
                self.kern_count,
                self.kernels_initializer,
                self.biases_initializer,
                self.kernels_optimizer,
                self.biases_optimizer,
            ),
            self.activation.to_layer(),
            self.dropout,
        )
    }
}