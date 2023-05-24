use serde::{Serialize, Deserialize};

use super::{ModelBuilder, conv_network_model::{ConvNetworkModelBuilder, ConvNetworkModel}, full_dense_layer_model::FullDenseLayerModel};

pub struct NetworkModelBuilder {
    pub model: NetworkModel,
    pub parent: ModelBuilder
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NetworkModel {
    pub layers: Vec<NetworkLayerModels>
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NetworkLayerModels {
    Convolution(ConvNetworkModel),
    FullDense(FullDenseLayerModel)
}

impl NetworkModelBuilder {
    pub fn new(parent: ModelBuilder) -> Self {
        Self {
            model: NetworkModel { layers: Vec::new() },
            parent
        }
    }

    pub fn conv_network(self) -> ConvNetworkModelBuilder {
        ConvNetworkModelBuilder::new(self)
    }

    pub fn accept_conv_network(mut self, layer: ConvNetworkModel) -> Self {
        self.model.layers.push(NetworkLayerModels::Convolution(layer));
        self
    }

    pub fn accept_full_dense(mut self, layer: FullDenseLayerModel) -> Self {
        self.model.layers.push(NetworkLayerModels::FullDense(layer));
        self
    }

    pub fn end(self) -> ModelBuilder {
        self.parent.accept_neural_network(self.model)
    }
}
