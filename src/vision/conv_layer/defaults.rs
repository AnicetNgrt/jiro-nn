use crate::vision::{
    conv_initializers::ConvInitializers,
    conv_optimizer::{conv_sgd, ConvOptimizers},
};

pub fn default_biases_initializer() -> ConvInitializers {
    ConvInitializers::Zeros
}

pub fn default_kernels_initializer() -> ConvInitializers {
    ConvInitializers::GlorotUniform
}

pub fn default_biases_optimizer() -> ConvOptimizers {
    conv_sgd()
}

pub fn default_kernels_optimizer() -> ConvOptimizers {
    conv_sgd()
}
