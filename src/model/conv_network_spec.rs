use serde::{Deserialize, Serialize};

use crate::vision::conv_network::{ConvNetwork, ConvNetworkLayer};

use super::full_conv_layer_spec::FullConvLayerSpec;

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ConvNetworkSpec {
    pub layers: Vec<ConvLayerSpecTypes>,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize,
}

impl ConvNetworkSpec {
    pub fn to_layer(self) -> ConvNetwork {
        let mut conv_layers: Vec<Box<dyn ConvNetworkLayer>> = vec![];
        let mut nchans = self.in_channels;
        for conv_layer_spec in self.layers.clone() {
            let (out_chans, layer) = conv_layer_spec.to_conv_layer(nchans);
            nchans = out_chans;
            conv_layers.push(layer);
        }
        ConvNetwork::new(conv_layers, self.in_channels)
    }

    pub fn out_size(&self, prev_out_size: usize) -> usize {
        let nrows = ((prev_out_size / self.in_channels) as f64).sqrt() as usize;
        let mut channels = self.in_channels;
        let mut image_size = (nrows, nrows);

        for layer in self.layers.clone() {
            let sizes = layer.out_size(image_size);
            image_size = sizes.1;
            channels = sizes.0;
        }

        channels * image_size.0 * image_size.1
    }
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum ConvLayerSpecTypes {
    Full(FullConvLayerSpec),
}

impl ConvLayerSpecTypes {
    pub fn out_size(&self, image_size: (usize, usize)) -> (usize, (usize, usize)) {
        match self {
            ConvLayerSpecTypes::Full(spec) => spec.out_size(image_size),
        }
    }

    pub fn to_conv_layer(self, nchans: usize) -> (usize, Box<dyn ConvNetworkLayer>) {
        match self {
            ConvLayerSpecTypes::Full(spec) => {
                (spec.conv.kern_count, Box::new(spec.to_conv_layer(nchans)))
            }
        }
    }
}
