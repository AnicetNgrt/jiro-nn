
use serde::{Serialize, Deserialize};

use crate::{network::NetworkLayer, vision::conv_network::ConvNetwork};

use super::{full_dense_conv_layer_model::{FullDenseConvLayerModel, FullDenseConvLayerModelBuilder}, network_model::NetworkModelBuilder, full_direct_conv_layer_model::FullDirectConvLayerModel};

pub struct ConvNetworkModelBuilder {
    pub model: ConvNetworkModel,
    parent: NetworkModelBuilder,
}

impl ConvNetworkModelBuilder {
    pub fn new(parent: NetworkModelBuilder, in_channels: usize) -> Self {
        Self { 
            model: ConvNetworkModel { layers: vec![], in_channels },
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

    pub fn accept_full_direct(mut self, model: FullDirectConvLayerModel) -> Self {
        self.model.layers.push(ConvNetworkLayerModels::FullDirectConv(model));
        self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConvNetworkModel {
    pub in_channels: usize,
    pub layers: Vec<ConvNetworkLayerModels>,
}

impl ConvNetworkModel {
    pub fn to_layer(&self, in_dims: usize) -> (usize, Box<dyn NetworkLayer>) {
        let mut layers = vec![];
        let mut in_channels = self.in_channels;
        let mut in_img_dims = (in_dims as f64 / in_channels as f64).sqrt() as usize;

        for layer_spec in self.layers {
            let (out_img_dims, out_channels, conv_layer) = layer_spec
                .to_conv_layer(in_img_dims, in_channels);

            in_img_dims = out_img_dims;
            in_channels = out_channels;
        }

        let network_layer = ConvNetwork::new(layers, self.in_channels);
        (in_img_dims * in_img_dims * in_channels, Box::new(network_layer))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ConvNetworkLayerModels {
    FullDenseConv(FullDenseConvLayerModel),
    FullDirectConv(FullDirectConvLayerModel),
    AvgPooling {
        kernel_size: usize,
    },
    
}