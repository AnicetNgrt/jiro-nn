use serde::{Serialize, Deserialize};

use crate::network::{Network, NetworkLayer};

use super::{ModelBuilder, conv_network_model::{ConvNetworkModelBuilder, ConvNetworkModel}, full_dense_layer_model::{FullDenseLayerModel, FullDenseLayerModelBuilder}};

pub struct NetworkModelBuilder {
    pub model: NetworkModel,
    pub parent: ModelBuilder
}

impl NetworkModelBuilder {
    pub fn new(parent: ModelBuilder) -> Self {
        Self {
            model: NetworkModel { layers: Vec::new() },
            parent
        }
    }

    pub fn conv_network(self, in_channels: usize) -> ConvNetworkModelBuilder {
        ConvNetworkModelBuilder::new(self, in_channels)
    }

    pub fn accept_conv_network(mut self, layer: ConvNetworkModel) -> Self {
        self.model.layers.push(NetworkLayerModels::Convolution(layer));
        self
    }

    pub fn full_dense(self, size: usize) -> FullDenseLayerModelBuilder {
        FullDenseLayerModelBuilder::new(self, size)
    }

    pub fn accept_full_dense(mut self, layer: FullDenseLayerModel) -> Self {
        self.model.layers.push(NetworkLayerModels::FullDense(layer));
        self
    }

    pub fn end(self) -> ModelBuilder {
        self.parent.accept_neural_network(self.model)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NetworkModel {
    pub layers: Vec<NetworkLayerModels>
}

impl NetworkModel {
    pub fn to_network(self, mut in_dims: usize) -> Network {
        let mut layers = vec![];
        for layer_spec in self.layers.into_iter() {
            let (out_dims, layer) = layer_spec.to_layer(in_dims);
            in_dims = out_dims;
            layers.push(layer);
        }
        Network::new(layers)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NetworkLayerModels {
    Convolution(ConvNetworkModel),
    FullDense(FullDenseLayerModel)
}

impl NetworkLayerModels {
    pub fn to_layer(self, in_dims: usize) -> (usize, Box<dyn NetworkLayer>) {
        match self {
            Self::Convolution(network) => network.to_layer(in_dims),
            Self::FullDense(layer) => layer.to_layer(in_dims)
        }
    }
}