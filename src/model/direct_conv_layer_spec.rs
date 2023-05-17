use crate::vision::{
    conv_initializers::ConvInitializers,
    conv_layer::{
        defaults::{
            default_biases_initializer, default_biases_optimizer, default_kernels_initializer,
            default_kernels_optimizer,
        },
        direct_conv_layer::DirectConvLayer,
    },
    conv_optimizer::ConvOptimizers,
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct DirectConvLayerSpec {
    pub kern_size: (usize, usize),
    pub kernels_optimizer: ConvOptimizers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_initializer: ConvInitializers,
    pub biases_initializer: ConvInitializers,
}

impl DirectConvLayerSpec {
    pub fn out_size(&self, image_size: (usize, usize), in_channels: usize) -> (usize, (usize, usize)) {
        (
            in_channels,
            (
                (self.kern_size.0.abs_diff(image_size.0) + 1),
                (self.kern_size.1.abs_diff(image_size.1) + 1),
            ),
        )
    }

    pub fn to_conv_layer(self, nchans: usize) -> DirectConvLayer {
        DirectConvLayer::new(
            self.kern_size.0,
            self.kern_size.1,
            nchans,
            self.kernels_initializer,
            self.biases_initializer,
            self.kernels_optimizer,
            self.biases_optimizer,
        )
    }

    pub fn from_options(options: &[DirectConvLayerOptions]) -> DirectConvLayerSpec {
        let mut spec = DirectConvLayerSpec::default();
        for option in options {
            match option {
                DirectConvLayerOptions::DCvKernelSize(rows, cols) => {
                    spec.kern_size = (rows.clone(), cols.clone())
                }
                DirectConvLayerOptions::DCvKernelsOptimizer(kernels_optimizer) => {
                    spec.kernels_optimizer = kernels_optimizer.clone()
                }
                DirectConvLayerOptions::DCvBiasesOptimizer(biases_optimizer) => {
                    spec.biases_optimizer = biases_optimizer.clone()
                }
                DirectConvLayerOptions::DCvKernelsInitializer(kernels_initializer) => {
                    spec.kernels_initializer = kernels_initializer.clone()
                }
                DirectConvLayerOptions::DCvBiasesInitializer(biases_initializer) => {
                    spec.biases_initializer = biases_initializer.clone()
                }
                DirectConvLayerOptions::DCvOptimizer(global_optimizer) => {
                    spec.kernels_optimizer = global_optimizer.clone();
                    spec.biases_optimizer = global_optimizer.clone();
                }
            }
        }
        spec
    }

    pub fn default() -> DirectConvLayerSpec {
        DirectConvLayerSpec {
            kernels_optimizer: default_kernels_optimizer(),
            biases_optimizer: default_biases_optimizer(),
            kernels_initializer: default_kernels_initializer(),
            biases_initializer: default_biases_initializer(),
            kern_size: (3, 3),
        }
    }
}

pub enum DirectConvLayerOptions {
    DCvKernelSize(usize, usize),
    DCvKernelsOptimizer(ConvOptimizers),
    DCvBiasesOptimizer(ConvOptimizers),
    DCvOptimizer(ConvOptimizers),
    DCvKernelsInitializer(ConvInitializers),
    DCvBiasesInitializer(ConvInitializers),
}
