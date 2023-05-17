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
pub struct DenseConvLayerSpec {
    pub kern_size: (usize, usize),
    pub kern_count: usize,
    pub kernels_optimizer: ConvOptimizers,
    pub biases_optimizer: ConvOptimizers,
    pub kernels_initializer: ConvInitializers,
    pub biases_initializer: ConvInitializers,
}

impl DenseConvLayerSpec {
    pub fn out_size(&self, image_size: (usize, usize)) -> (usize, (usize, usize)) {
        (
            self.kern_count,
            (
                (self.kern_size.0.abs_diff(image_size.0) + 1),
                (self.kern_size.1.abs_diff(image_size.1) + 1),
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

    pub fn from_options(options: &[DenseConvLayerOptions]) -> DenseConvLayerSpec {
        let mut spec = DenseConvLayerSpec::default();
        for option in options {
            match option {
                DenseConvLayerOptions::KernelCount(count) => {
                    spec.kern_count = count.clone()
                }
                DenseConvLayerOptions::KernelSize(rows, cols) => {
                    spec.kern_size = (rows.clone(), cols.clone())
                }
                DenseConvLayerOptions::KernelsOptimizer(kernels_optimizer) => {
                    spec.kernels_optimizer = kernels_optimizer.clone()
                }
                DenseConvLayerOptions::ConvBiasesOptimizer(biases_optimizer) => {
                    spec.biases_optimizer = biases_optimizer.clone()
                }
                DenseConvLayerOptions::KernelsInitializer(kernels_initializer) => {
                    spec.kernels_initializer = kernels_initializer.clone()
                }
                DenseConvLayerOptions::ConvBiasesInitializer(biases_initializer) => {
                    spec.biases_initializer = biases_initializer.clone()
                }
                DenseConvLayerOptions::ConvOptimizer(global_optimizer) => {
                    spec.kernels_optimizer = global_optimizer.clone();
                    spec.biases_optimizer = global_optimizer.clone();
                }
            }
        }
        spec
    }

    pub fn default() -> DenseConvLayerSpec {
        DenseConvLayerSpec {
            kernels_optimizer: default_kernels_optimizer(),
            biases_optimizer: default_biases_optimizer(),
            kernels_initializer: default_kernels_initializer(),
            biases_initializer: default_biases_initializer(),
            kern_size: (3, 3),
            kern_count: 1,
        }
    }
}

pub enum DenseConvLayerOptions {
    KernelCount(usize),
    KernelSize(usize, usize),
    KernelsOptimizer(ConvOptimizers),
    ConvBiasesOptimizer(ConvOptimizers),
    ConvOptimizer(ConvOptimizers),
    KernelsInitializer(ConvInitializers),
    ConvBiasesInitializer(ConvInitializers),
}
