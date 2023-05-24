use super::full_dense_conv_layer_model::FullDenseConvLayerModel;

pub struct NetworkModel {
    pub layers: Vec<NetworkLayerModels>
}

pub enum NetworkLayerModels {
    Convolution(ConvNetworkModel),
}

impl NetworkModel {
    pub fn new() -> Self {
        Self { 
            layers: Vec::new(),
        }
    }

    pub fn conv_network(self) -> ConvNetworkModel {
        ConvNetworkModel::new(self)
    }
}

pub struct ConvNetworkModel {
    pub layers: Vec<ConvNetworkLayerModels>,
    parent: NetworkModel,
}

pub enum ConvNetworkLayerModels {
    FullDenseConv(FullDenseConvLayerModel),
    AvgPooling {
        kernel_size: usize,
    },
    
}

impl ConvNetworkModel {
    pub fn new(parent: NetworkModel) -> Self {
        Self { 
            layers: Vec::new(),
            parent,
        }
    }

    pub fn end(self) -> NetworkModel {
        self.parent
    }

    pub fn full_dense(self, kernels_count: usize, kernels_size: usize) -> FullDenseConvLayerModel {
        FullDenseConvLayerModel::new(self, kernels_count, kernels_size)
    }

    pub fn accept_full_dense(mut self, model: FullDenseConvLayerModel) -> Self {
        self.layers.push(ConvNetworkLayerModels::FullDenseConv(model));
        self
    }

    pub fn avg_pooling(self, kernel_size: usize) -> Self {
        Self {
            layers: self.layers,
            parent: self.parent,
        }
    }
}