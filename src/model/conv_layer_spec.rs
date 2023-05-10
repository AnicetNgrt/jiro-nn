use crate::vision::{
    conv_initializers::ConvInitializers,
    conv_layer::{
        defaults::{
            default_biases_initializer, default_biases_optimizer, default_kernels_initializer,
            default_kernels_optimizer,
        },
        dense_conv_layer::DenseConvLayer,
    },
    conv_optimizer::ConvOptimizers,
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ConvLayerSpec {
    pub kern_size: (usize, usize),
    pub kern_count: usize,
    pub kernels_optimizer: ConvOptimizers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_initializer: ConvInitializers,
    pub biases_initializer: ConvInitializers,
}

impl ConvLayerSpec {
    pub fn out_size(&self, image_size: (usize, usize)) -> (usize, (usize, usize)) {
        (
            self.kern_count,
            (
                (self.kern_size.0 - image_size.0 + 1),
                (self.kern_size.1 - image_size.1 + 1),
            ),
        )
    }

    pub fn to_conv_layer(self, nchans: usize) -> DenseConvLayer {
        DenseConvLayer::new(
            self.kern_size.0,
            self.kern_size.1,
            nchans,
            self.kern_count,
            self.kernels_initializer,
            self.biases_initializer,
            self.kernels_optimizer,
            self.biases_optimizer,
        )
    }

    pub fn from_options(options: &[ConvLayerOptions]) -> ConvLayerSpec {
        let mut spec = ConvLayerSpec::default();
        for option in options {
            match option {
                ConvLayerOptions::KernelSize(rows, cols) => {
                    spec.kern_size = (rows.clone(), cols.clone())
                }
                ConvLayerOptions::KernelsOptimizer(kernels_optimizer) => {
                    spec.kernels_optimizer = kernels_optimizer.clone()
                }
                ConvLayerOptions::BiasesOptimizer(biases_optimizer) => {
                    spec.biases_optimizer = biases_optimizer.clone()
                }
                ConvLayerOptions::KernelsInitializer(kernels_initializer) => {
                    spec.kernels_initializer = kernels_initializer.clone()
                }
                ConvLayerOptions::BiasesInitializer(biases_initializer) => {
                    spec.biases_initializer = biases_initializer.clone()
                }
                ConvLayerOptions::Optimizer(global_optimizer) => {
                    spec.kernels_optimizer = global_optimizer.clone();
                    spec.biases_optimizer = global_optimizer.clone();
                }
            }
        }
        spec
    }

    pub fn default() -> ConvLayerSpec {
        ConvLayerSpec {
            kernels_optimizer: default_kernels_optimizer(),
            biases_optimizer: default_biases_optimizer(),
            kernels_initializer: default_kernels_initializer(),
            biases_initializer: default_biases_initializer(),
            kern_size: (3, 3),
            kern_count: 1,
        }
    }
}

pub enum ConvLayerOptions {
    KernelSize(usize, usize),
    KernelsOptimizer(ConvOptimizers),
    BiasesOptimizer(ConvOptimizers),
    Optimizer(ConvOptimizers),
    KernelsInitializer(ConvInitializers),
    BiasesInitializer(ConvInitializers),
}
