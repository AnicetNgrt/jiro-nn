#![doc = include_str!("../README.md")]

#[macro_use]
extern crate assert_float_eq;

pub mod activation;
pub mod benchmarking;
pub mod dataset;
pub mod datatable;
pub mod initializers;
pub mod layer;
pub mod learning_rate;
pub mod loss;
pub mod model;
pub mod network;
pub mod optimizer;
pub mod pipelines;
pub mod vec_utils;
pub mod linalg;
pub mod trainers;
pub mod vision;
