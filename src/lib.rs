use activation::Activation;
use layer::dense_layer::DenseLayer;
use layer::full_layer::FullLayer;
use layer::hidden_layer::HiddenLayer;
use layer::skip_layer::SkipLayer;
use network::Network;

pub mod activation;
pub mod layer;
pub mod loss;
pub mod network;

// Neural network with I inputs and J outputs and no hidden layers
pub fn nn_h0<const I: usize, const J: usize>(activation: Activation) -> Network<I, J> {
    Network::new(Box::new(FullLayer::<I, J>::new(
        DenseLayer::new(),
        activation.to_layer(),
    )))
}

// Neural network with I inputs and J outputs and 1 hidden layer of size H
pub fn nn_h1<const I: usize, const H: usize, const J: usize>(
    activations: Vec<Activation>,
) -> Network<I, J> {
    let layer0 = FullLayer::<I, H>::new(
        DenseLayer::new(),
        activations[0 % activations.len()].to_layer(),
    );
    let layer1 = SkipLayer::<H>;
    let layer2 = FullLayer::<H, J>::new(
        DenseLayer::new(),
        activations[1 % activations.len()].to_layer(),
    );
    let global = HiddenLayer::new(layer0, layer1, layer2);

    Network::new(Box::new(global))
}

// Neural network with I inputs and J outputs and 2 hidden layers of sizes H0 & H1
pub fn nn_h2<const I: usize, const H0: usize, const H1: usize, const J: usize>(
    activations: Vec<Activation>,
) -> Network<I, J> {
    let layer0 = FullLayer::<I, H0>::new(
        DenseLayer::new(),
        activations[0 % activations.len()].to_layer(),
    );
    let layer1 = FullLayer::<H0, H1>::new(
        DenseLayer::new(),
        activations[1 % activations.len()].to_layer(),
    );
    let layer2 = FullLayer::<H1, J>::new(
        DenseLayer::new(),
        activations[2 % activations.len()].to_layer(),
    );
    let global = HiddenLayer::new(layer0, layer1, layer2);

    Network::new(Box::new(global))
}
