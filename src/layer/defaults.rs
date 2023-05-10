use crate::{
    initializers::Initializers,
    optimizer::{sgd, Optimizers},
};

pub fn default_biases_initializer() -> Initializers {
    Initializers::Zeros
}

pub fn default_weights_initializer() -> Initializers {
    Initializers::GlorotUniform
}

pub fn default_biases_optimizer() -> Optimizers {
    sgd()
}

pub fn default_weights_optimizer() -> Optimizers {
    sgd()
}
