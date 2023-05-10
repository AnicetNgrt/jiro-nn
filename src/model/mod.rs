use std::fs::File;
use std::io::{Read, Write};

use polars::prelude::NamedFrom;
use polars::series::Series;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::datatable::DataTable;
use crate::linalg::Scalar;
use crate::loss::Losses;
use crate::network::{Network, NetworkLayer};
use crate::trainers::Trainers;
use crate::{dataset::Dataset};

use self::conv_network_spec::ConvNetworkSpec;
use self::full_layer_spec::FullLayerSpec;

pub mod full_layer_spec;
pub mod conv_network_spec;
pub mod conv_layer_spec;

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Model {
    pub epochs: usize,
    pub loss: Losses,
    pub hidden_layers: Vec<LayerSpecTypes>,
    pub final_layer: LayerSpecTypes,
    pub batch_size: Option<usize>,
    pub trainer: Trainers,
    pub dataset: Dataset,
}

impl Model {
    pub fn from_json_file<S: AsRef<str>>(file_name: S) -> Model {
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

    pub fn from_json_string<S: AsRef<str>>(json: S) -> Model {
        let spec: Model = serde_json::from_str(json.as_ref()).unwrap();
        spec
    }

    pub fn hashed_repr(&self) -> String {
        let json = serde_json::to_string(&self).unwrap();
        let hash = Sha256::digest(json.as_bytes());
        let hash_string = format!("{:x}", hash);
        hash_string
    }

    pub fn with_new_dataset(&mut self, dataset: Dataset) -> Model {
        let mut spec = self.clone();
        spec.dataset = dataset;
        spec
    }

    pub fn train_epoch(&self, epoch: usize, network: &mut Network, train_data: &DataTable, id_column: &str) -> Scalar {
        let (train_x_table, train_y_table) =
            train_data.random_order_in_out(&self.dataset.out_features_names());

        let train_x = train_x_table.drop_column(id_column).to_vectors();
        let train_y = train_y_table.to_vectors();

        let train_loss = network.train(
            epoch,
            &train_x,
            &train_y,
            &self.loss.to_loss(),
            self.batch_size.unwrap_or(train_x.len()),
        );

        train_loss
    }

    fn compute_network_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.dataset.in_features_names().len()];
        let mut prev_out_size = sizes[0];
        for layer_spec in self.hidden_layers.iter() {
            let size = layer_spec.compute_out_size(prev_out_size);
            sizes.push(size);
            prev_out_size = size;
        }
        sizes.push(self.dataset.out_features_names().len());
        sizes
    }

    pub fn to_network(&self) -> Network {
        let sizes = self.compute_network_sizes();

        let mut layers: Vec<Box<dyn NetworkLayer>> = vec![];

        for i in 0..sizes.len() - 1 {
            let layer_spec = if i == sizes.len() - 2 {
                self.final_layer.clone()
            } else {
                self.hidden_layers[i].clone()
            };
            let in_size = sizes[i];
            layers.push(
                layer_spec.to_network_layer(in_size)
            );
        }

        Network::new(layers)
    }

    /// Uses the model's dataset specification to label the prediction's columns and convert it all to a `DataTable` spreadsheet.
    pub fn preds_to_table(&self, preds: Vec<Vec<Scalar>>) -> DataTable {
        let mut table = DataTable::new_empty();
        let names = self.dataset.out_features_names();
        let mut preds_columns: Vec<Vec<Scalar>> = Vec::new();

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

    /// The `from_options` method is a constructor function for creating a `Model` object from a list of `ModelOptions`.
    ///
    /// See the `ModelOptions` enum for more information.
    pub fn from_options(options: &[ModelOptions]) -> Model {
        let mut spec = Model::default();
        for option in options {
            match option {
                ModelOptions::Epochs(epochs) => spec.epochs = epochs.clone(),
                ModelOptions::Loss(loss) => spec.loss = loss.clone(),
                ModelOptions::HiddenLayers(hidden_layers) => {
                    spec.hidden_layers = hidden_layers.to_vec()
                }
                ModelOptions::FinalLayer(final_layer) => spec.final_layer = final_layer.clone(),
                ModelOptions::BatchSize(batch_size) => spec.batch_size = Some(batch_size.clone()),
                ModelOptions::Dataset(dataset) => spec.dataset = dataset.clone(),
                ModelOptions::Trainer(trainer) => spec.trainer = trainer.clone()
            }
        }
        spec
    }

    pub fn default() -> Model {
        Model {
            epochs: 100,
            loss: Losses::MSE,
            hidden_layers: vec![],
            final_layer: LayerSpecTypes::Full(FullLayerSpec::default()),
            batch_size: None,
            trainer: Trainers::KFolds(10),
            dataset: Dataset::default(),
        }
    }
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum LayerSpecTypes {
    Full(FullLayerSpec),
    ConvNetwork(ConvNetworkSpec),
}

impl LayerSpecTypes {
    pub fn compute_out_size(&self, prev_out_size: usize) -> usize {
        match self {
            LayerSpecTypes::Full(layer_spec) => {
                layer_spec.out_size
            }
            LayerSpecTypes::ConvNetwork(layer_spec) => {
                layer_spec.out_size(prev_out_size)
            }
        }
    }

    pub fn to_network_layer(self, in_size: usize) -> Box<dyn NetworkLayer> {
        match self {
            LayerSpecTypes::Full(layer_spec) => {
                let layer = layer_spec.to_layer(in_size);
                Box::new(layer) as Box<dyn NetworkLayer>
            }
            LayerSpecTypes::ConvNetwork(layer_spec) => {
                let layer = layer_spec.to_layer();
                Box::new(layer) as Box<dyn NetworkLayer>
            }
        }
    } 
}

/// Options to be used in order to build a `Model` struct with the `from_options` method.
///
/// **Required options**:
///
/// - `HiddenLayers`: The hidden layers of the model (an array of `FullLayerSpec` objects).
/// - `FinalLayer`: The final layer of the model (a `LayerSpecTypes` enum).
/// - `Dataset`: The dataset to be used for training (a `Dataset` object).
///
/// **Optional options**:
///
/// - `Epochs`: The number of epochs to train for. If ommited, defaults to `100`.
/// - `Loss`: The loss function to be used (a variant of the `Losses` enum). If ommited, defaults to `Losses::MSE`.
/// - `BatchSize`: The batch size to be used during training. If ommited, defaults to the size of the dataset.
/// - `Folds`: The number of folds to be used during cross-validation. If ommited, defaults to `10`.
pub enum ModelOptions<'a> {
    /// The number of epochs to train for. If ommited, defaults to `100`.
    Epochs(usize),
    /// The loss function to be used (a variant of the `Losses` enum). If ommited, defaults to `Losses::MSE`.
    Loss(Losses),
    /// The hidden layers of the model (an array of `LayerSpecTypes` enums).
    HiddenLayers(&'a [LayerSpecTypes]),
    /// The final layer of the model (a `LayerSpecTypes` enum).
    FinalLayer(LayerSpecTypes),
    /// The batch size to be used during training. If ommited, defaults to the size of the dataset.
    BatchSize(usize),
    /// The dataset to be used for training (a `Dataset` object).
    Dataset(Dataset),
    /// The trainer to use (a variant of the `Trainers` enum). If ommited, defaults to `Trainers::KFolds(10)`.
    Trainer(Trainers),
}
