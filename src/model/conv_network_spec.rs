use serde::{Deserialize, Serialize};

use crate::vision::conv_network::{ConvNetwork, ConvNetworkLayer};

use super::conv_layer_spec::ConvLayerSpec;

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ConvNetworkSpec {
    pub layers: Vec<ConvLayerSpec>,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize
}

impl ConvNetworkSpec {
    pub fn to_layer(self) -> ConvNetwork {
        let mut conv_layers: Vec<Box<dyn ConvNetworkLayer>> = vec![];
        let mut nchans = self.in_channels;
        for conv_layer_spec in self.layers.clone() {
            let in_chans = nchans;
            nchans = conv_layer_spec.kern_count;
            let layer = conv_layer_spec.to_conv_layer(in_chans);
            conv_layers.push(Box::new(layer));
        }
        ConvNetwork::new(
            conv_layers,
            self.in_channels
        )
    }

    pub fn out_size(&self, prev_out_size: usize) -> usize {
        (prev_out_size / self.in_channels) * self.layers.last().unwrap().kern_count
    }
}