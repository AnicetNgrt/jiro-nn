
use serde::{Serialize, Deserialize};

use super::{full_dense_conv_layer_model::{FullDenseConvLayerModel, FullDenseConvLayerModelBuilder}, network_model::NetworkModelBuilder};

pub struct ConvNetworkModelBuilder {
    pub model: ConvNetworkModel,
    parent: NetworkModelBuilder,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConvNetworkModel {
    pub layers: Vec<ConvNetworkLayerModels>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ConvNetworkLayerModels {
    FullDenseConv(FullDenseConvLayerModel),
    AvgPooling {
        kernel_size: usize,
    },
    
}

impl ConvNetworkModelBuilder {
    pub fn new(parent: NetworkModelBuilder) -> Self {
        Self { 
            model: ConvNetworkModel { layers: vec![] },
            parent,
        }
    }

    pub fn end(self) -> NetworkModelBuilder {
        self.parent.accept_conv_network(self.model)
    }

    pub fn full_dense(self, kernels_count: usize, kernels_size: usize) -> FullDenseConvLayerModelBuilder {
        FullDenseConvLayerModelBuilder::new(self, kernels_count, kernels_size)
    }

    pub fn avg_pooling(mut self, kernel_size: usize) -> Self {
        self.model.layers.push(ConvNetworkLayerModels::AvgPooling { kernel_size });
        self
    }

    pub fn accept_full_dense(mut self, model: FullDenseConvLayerModel) -> Self {
        self.model.layers.push(ConvNetworkLayerModels::FullDenseConv(model));
        self
    }
}