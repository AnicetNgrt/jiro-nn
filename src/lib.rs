use activation::Activation;
use layer::dense_layer::DenseLayer;
use layer::full_layer::{FullLayer};
use network::Network;

pub mod activation;
pub mod layer;
pub mod loss;
pub mod network;
pub mod optimizer;
pub mod datatable;
pub mod benchmarking;
pub mod stats_utils;

// Neural network with I inputs and J outputs and 2 hidden layers of sizes H0 & H1
pub fn nn(
    activations: Vec<Activation>,
    sizes: Vec<usize>
) -> Network {
    let mut layers = vec![];

    for i in 0..sizes.len()-1 {
        let in_size = sizes[i];
        let out_size = sizes[i+1];
        let layer = FullLayer::new(
            DenseLayer::new(in_size, out_size),
            activations[i % activations.len()].to_layer(),
        );
        layers.push(layer);
    }

    Network::new(layers, *sizes.first().unwrap(), *sizes.last().unwrap())
}