use activation_layer::ActivationLayer;
use dense_layer::DenseLayer;
use full_layer::FullLayer;
use network::Network;
use skip_layer::SkipLayer;
use with_hidden::WithHidden;

pub mod activation_layer;
pub mod with_hidden;
pub mod dense_layer;
pub mod full_layer;
pub mod layer;
pub mod mean_squared_error;
pub mod network;
pub mod tanh_activation;
pub mod skip_layer;
pub mod loss;

pub enum Activation {
    Tanh
}

impl Activation {
    pub fn to_layer<const I: usize>(&self) -> ActivationLayer<I> {
        match self {
            Activation::Tanh => tanh_activation::new(),
        }
    }
}

// Neural network with I inputs and J outputs and no hidden layers
pub fn nn_h0<const I: usize, const J: usize>(activation: Activation) -> Network<I, J> {
    Network::new(Box::new(FullLayer::<I, J>::new(
        DenseLayer::new(),
        activation.to_layer()
    )))
}

// Neural network with I inputs and J outputs and 1 hidden layer of size H
pub fn nn_h1<const I: usize, const H: usize, const J: usize>(activations: Vec<Activation>) -> Network<I, J> {
    let layer0 = FullLayer::<I, H>::new(
        DenseLayer::new(),
        activations[0 % activations.len()].to_layer()
    );
    let layer1 = SkipLayer::<H>;
    let layer2 = FullLayer::<H, J>::new(
        DenseLayer::new(),
        activations[1 % activations.len()].to_layer()
    );
    let global = WithHidden::new(
        layer0,
        layer1,
        layer2
    );

    Network::new(Box::new(global))
}

// Neural network with I inputs and J outputs and 2 hidden layers of sizes H0 & H1
pub fn nn_h2<const I: usize, const H0: usize, const H1: usize, const J: usize>(activations: Vec<Activation>) -> Network<I, J> {
    let layer0 = FullLayer::<I, H0>::new(
        DenseLayer::new(),
        activations[0 % activations.len()].to_layer()
    );
    let layer1 = FullLayer::<H0, H1>::new(
        DenseLayer::new(),
        activations[1 % activations.len()].to_layer()
    );
    let layer2 = FullLayer::<H1, J>::new(
        DenseLayer::new(),
        activations[2 % activations.len()].to_layer()
    );
    let global = WithHidden::new(
        layer0,
        layer1,
        layer2
    );

    Network::new(Box::new(global))
}