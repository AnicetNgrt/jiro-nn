use serde::{Serialize, Deserialize};

use crate::network::{Network, NetworkLayer};

use super::{ModelBuilder, conv_network_model::{ConvNetworkModelBuilder, ConvNetworkModel}, full_dense_layer_model::{FullDenseLayerModel, FullDenseLayerModelBuilder}};

pub struct NetworkModelBuilder {
    pub model: NetworkModel,
    pub parent: Option<ModelBuilder>
}

impl NetworkModelBuilder {
    pub fn new() -> Self {
        Self {
            model: NetworkModel { layers: Vec::new() },
            parent: None
        }
    }

    pub fn set_parent(mut self, parent: ModelBuilder) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn conv_network(self, in_channels: usize) -> ConvNetworkModelBuilder {
        ConvNetworkModelBuilder::new(self, in_channels)
    }

    pub(crate) fn accept_conv_network(mut self, layer: ConvNetworkModel) -> Self {
        self.model.layers.push(NetworkLayerModels::Convolution(layer));
        self
    }

    pub fn full_dense(self, size: usize) -> FullDenseLayerModelBuilder {
        FullDenseLayerModelBuilder::new(self, size)
    }

    pub(crate) fn accept_full_dense(mut self, layer: FullDenseLayerModel) -> Self {
        self.model.layers.push(NetworkLayerModels::FullDense(layer));
        self
    }

    pub fn end(self) -> ModelBuilder {
        match self.parent {
            Some(parent) => parent.accept_neural_network(self.model),
            None => panic!("No parent model builder set")
        }
    }

    pub fn build(self) -> NetworkModel {
        self.model
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