use std::fs::File;
use std::io::{Read, Write};

use polars::prelude::NamedFrom;
use polars::series::Series;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::datatable::DataTable;
use crate::initializers::{Initializers};
use crate::layer::dense_layer::{DenseLayer, default_biases_initializer, default_weights_initializer, default_weights_optimizer, default_biases_optimizer};
use crate::layer::full_layer::FullLayer;
use crate::loss::Losses;
use crate::network::Network;
use crate::{activation::Activation, dataset::Dataset, optimizer::Optimizers};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ModelSpec {
    pub epochs: usize,
    pub loss: Losses,
    pub hidden_layers: Vec<LayerSpec>,
    pub final_layer: LayerSpec,
    pub batch_size: Option<usize>,
    pub folds: usize,
    pub dataset: Dataset,
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct LayerSpec {
    #[serde(default)]
    pub out_size: usize,
    pub activation: Activation,
    pub dropout: Option<f64>,
    #[serde(default="default_weights_optimizer")]
    pub weights_optimizer: Optimizers,
    #[serde(default="default_biases_optimizer")]
    pub biases_optimizer: Optimizers,
    #[serde(default="default_biases_initializer")]
    pub biases_initializer: Initializers,
    #[serde(default="default_weights_initializer")]
    pub weights_initializer: Initializers,
}

pub enum ModelOptions<'a> {
    Epochs(usize),
    Loss(Losses),
    HiddenLayers(&'a [LayerSpec]),
    FinalLayer(LayerSpec),
    BatchSize(Option<usize>),
    Folds(usize),
    Dataset(Dataset),
}

impl ModelSpec {
    pub fn from_json_file<S: AsRef<str>>(file_name: S) -> ModelSpec {
        let mut file = File::open(file_name.as_ref()).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        Self::from_json_string(&contents)
    }

    pub fn to_json_file<S: AsRef<str>>(&self, file_name: S) {
        let json = serde_json::to_string_pretty(&self).unwrap();
        let mut file = File::create(file_name.as_ref()).unwrap();
        file.write_all(json.as_bytes()).unwrap();
    }

    pub fn from_json_string<S: AsRef<str>>(json: S) -> ModelSpec {
        let spec: ModelSpec = serde_json::from_str(json.as_ref()).unwrap();
        spec
    }

    pub fn hashed_repr(&self) -> String {
        let json = serde_json::to_string(&self).unwrap();
        let hash = Sha256::digest(json.as_bytes());
        let hash_string = format!("{:x}", hash);
        hash_string
    }

    pub fn with_new_dataset(&self, dataset: Dataset) -> ModelSpec {
        let mut spec = self.clone();
        spec.dataset = dataset;
        spec
    }

    pub fn to_network(&self) -> Network {
        let mut sizes = vec![self.dataset.in_features_names().len()];
        sizes.append(&mut self.hidden_layers.iter().map(|l| l.out_size).collect());
        sizes.push(self.dataset.out_features_names().len());
        
        let mut layers = vec![];

        for i in 0..sizes.len() - 1 {
            let layer_spec = if i == sizes.len() - 2 {
                &self.final_layer
            } else {
                &self.hidden_layers[i]
            };
            let in_size = sizes[i];
            let out_size = sizes[i + 1];
            let layer = FullLayer::new(
                DenseLayer::new(
                    in_size,
                    out_size,
                    layer_spec.weights_optimizer.clone(),
                    layer_spec.biases_optimizer.clone(),
                    layer_spec.weights_initializer.clone(),
                    layer_spec.biases_initializer.clone()
                ),
                layer_spec.activation.to_layer(),
                layer_spec.dropout
            );
            layers.push(layer);
        }

        Network::new(layers, *sizes.first().unwrap(), *sizes.last().unwrap())
    }

    pub fn preds_to_table(&self, preds: Vec<Vec<f64>>) -> DataTable {
        let mut table = DataTable::new_empty();
        let names = self.dataset.out_features_names();
        let mut preds_columns: Vec<Vec<f64>> = Vec::new();

        // inverse the transpose
        for i in 0..preds[0].len() {
            let mut column = vec![];
            for j in 0..preds.len() {
                column.push(preds[j][i]);
            }
            preds_columns.push(column);
        }

        for (i, pred) in preds.iter().enumerate() {
            table = table.append_column(Series::new(&format!("pred_{}", names[i]), pred.clone()));
        }
        table
    }

    /// The `from_options` method is a constructor function for creating a `ModelSpec` object from a list of `ModelOptions`.
    /// 
    /// The possible options are:
    /// 
    /// - `Epochs`: The number of epochs to train for (an integer).
    /// - `Loss`: The loss function to be used (a variant of the `Losses` enum).
    /// - `HiddenLayers`: The hidden layers of the model (an array of `LayerSpec` objects).
    /// - `FinalLayer`: The final layer of the model (a `LayerSpec` object).
    /// - `BatchSize`: The batch size to be used during training (an optional integer).
    /// - `Folds`: The number of folds to be used during cross-validation (an integer).
    /// - `Dataset`: The dataset to be used for training (a variant of the `Dataset` enum).

    pub fn from_options(options: &[ModelOptions]) -> ModelSpec {
        let mut spec = ModelSpec::default();
        for option in options {
            match option {
                ModelOptions::Epochs(epochs) => spec.epochs = epochs.clone(),
                ModelOptions::Loss(loss) => spec.loss = loss.clone(),
                ModelOptions::HiddenLayers(hidden_layers) => spec.hidden_layers = hidden_layers.to_vec(),
                ModelOptions::FinalLayer(final_layer) => spec.final_layer = final_layer.clone(),
                ModelOptions::BatchSize(batch_size) => spec.batch_size = batch_size.clone(),
                ModelOptions::Folds(folds) => spec.folds = folds.clone(),
                ModelOptions::Dataset(dataset) => spec.dataset = dataset.clone(),
            }
        }
        spec
    }

    pub fn default() -> ModelSpec {
        ModelSpec {
            epochs: 100,
            loss: Losses::MSE,
            hidden_layers: vec![],
            final_layer: LayerSpec::default(),
            batch_size: None,
            folds: 5,
            dataset: Dataset::default(),
        }
    }
}

pub enum LayerOptions {
    OutSize(usize),
    Activation(Activation),
    Dropout(Option<f64>),
    WeightsOptimizer(Optimizers),
    BiasesOptimizer(Optimizers),
    WeightsInitializer(Initializers),
    BiasesInitializer(Initializers),
}

impl LayerSpec {

    /// The `from_options` method is a constructor function for creating a `LayerSpec` object from a list of `LayerOptions`.
    /// 
    /// The possible options are:
    /// 
    /// - `OutSize`: The size of the output layer (an integer).
    /// - `Activation`: The activation function to be used (a variant of the `Activation` enum).
    /// - `Dropout`: The dropout rate (an optional float).
    /// - `WeightsOptimizer`: The optimizer to be used for updating the weights (a variant of the `Optimizers` enum).
    /// - `BiasesOptimizer`: The optimizer to be used for updating the biases (a variant of the `Optimizers` enum).
    /// - `WeightsInitializer`: The initializer to be used for initializing the weights (a variant of the `Initializers` enum).
    /// - `BiasesInitializer`: The initializer to be used for initializing the biases (a variant of the `Initializers` enum).

    pub fn from_options(options: &[LayerOptions]) -> LayerSpec {
        let mut spec = LayerSpec::default();
        for option in options {
            match option {
                LayerOptions::OutSize(out_size) => spec.out_size = out_size.clone(),
                LayerOptions::Activation(activation) => spec.activation = activation.clone(),
                LayerOptions::Dropout(dropout) => spec.dropout = dropout.clone(),
                LayerOptions::WeightsOptimizer(weights_optimizer) => spec.weights_optimizer = weights_optimizer.clone(),
                LayerOptions::BiasesOptimizer(biases_optimizer) => spec.biases_optimizer = biases_optimizer.clone(),
                LayerOptions::WeightsInitializer(weights_initializer) => spec.weights_initializer = weights_initializer.clone(),
                LayerOptions::BiasesInitializer(biases_initializer) => spec.biases_initializer = biases_initializer.clone(),
            }
        }
        spec
    }

    pub fn default() -> LayerSpec {
        LayerSpec {
            out_size: 0,
            activation: Activation::Linear,
            dropout: None,
            weights_optimizer: default_weights_optimizer(),
            biases_optimizer: default_biases_optimizer(),
            weights_initializer: default_weights_initializer(),
            biases_initializer: default_biases_initializer(),
        }
    }
}