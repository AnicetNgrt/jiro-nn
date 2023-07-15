#![doc = include_str!("../README.md")]

#[macro_use]
extern crate assert_float_eq;

/// Monitoring performance of tasks and logging
pub mod monitor;
/// Activation functions and abstractions (sigmoid, relu, softmax...)
pub mod activation;
/// Model performance benchmarking utilities
pub mod benchmarking;
#[cfg(feature = "data")]
/// Dataset configuration (metadata, preprocessing flags...)
pub mod dataset;
#[cfg(feature = "data")]
/// Wrapper around dataframes libaries
pub mod datatable;
/// Parameters initializers and abstractions (uniform, glorot...)
pub mod initializers;
/// Layers and abstractions (dense, full...)
pub mod layer;
/// Learning rate schedulers and abstractions (constant, exponential...)
pub mod learning_rate;
/// Basic linear algebra backends and wrappers (matrix, vector, scalar...)
pub mod linalg;
/// Loss functions and abstractions (mse, crossentropy...)
pub mod loss;
/// Model configuration
pub mod model;
/// Neural network abstractions
pub mod network;
/// Optimizers and abstractions (sgd, adam...)
pub mod optimizer;
#[cfg(feature = "data")]
/// Preprocessing and pipelining utilities (normalization, one-hot encoding...)
pub mod preprocessing;
/// Training methodologies (k-fold, split...)
pub mod trainers;
/// Utilities for `Vec<Scalar>`, `Vec<Vec<Scalar>>`...
pub mod vec_utils;
/// Everything vision (CNNs, images operations & backends...)
pub mod vision;

pub mod ops;