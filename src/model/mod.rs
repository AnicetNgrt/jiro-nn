use std::fs::File;
use std::io::{Read, Write};

#[cfg(feature = "data")]
use polars::prelude::NamedFrom;
#[cfg(feature = "data")]
use polars::series::Series;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "data")]
use crate::dataset::Dataset;
#[cfg(feature = "data")]
use crate::datatable::DataTable;

use crate::linalg::Scalar;
use crate::loss::Losses;
use crate::network::{Network};

use self::network_model::{NetworkModel, NetworkModelBuilder};

pub mod network_model;
pub mod conv_network_model;
pub mod full_dense_layer_model;
pub mod full_direct_conv_layer_model;
pub mod full_dense_conv_layer_model;

pub struct ModelBuilder {
    pub model: Model
}

impl ModelBuilder {
    #[cfg(feature = "data")]
    pub fn new(dataset_config: Dataset) -> Self {
        Self {
            model: Model {
                dataset_config,
                loss: Losses::MSE,
                epochs: 100,
                batch_size: Some(32),
                network: None
            }
        }
    }

    #[cfg(not(feature = "data"))]
    pub fn new() -> Self {
        Self {
            model: Model {
                loss: Losses::MSE,
                epochs: 100,
                batch_size: Some(32),
                network: None
            }
        }
    }

    pub fn loss(self, loss: Losses) -> Self {
        Self {
            model: Model {
                loss,
                ..self.model
            },
            ..self
        }
    }

    pub fn epochs(self, epochs: usize) -> Self {
        Self {
            model: Model {
                epochs,
                ..self.model
            },
            ..self
        }
    }

    pub fn batch_size(self, batch_size: usize) -> Self {
        Self {
            model: Model {
                batch_size: Some(batch_size),
                ..self.model
            },
            ..self
        }
    }

    pub fn neural_network(self) -> NetworkModelBuilder {
        NetworkModelBuilder::new().set_parent(self)
    }

    pub(crate) fn accept_neural_network(mut self, network: NetworkModel) -> Self {
        self.model.network = Some(network);
        self
    }

    pub fn build(self) -> Model {
        self.model
    }
}

#[cfg(feature = "data")]
#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Model {
    pub epochs: usize,
    pub loss: Losses,
    pub batch_size: Option<usize>,
    pub dataset_config: Dataset,
    pub network: Option<NetworkModel>
}

#[cfg(not(feature = "data"))]
#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Model {
    pub epochs: usize,
    pub loss: Losses,
    pub batch_size: Option<usize>,
    pub network: Option<NetworkModel>
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
        let model: Model = serde_json::from_str(json.as_ref()).unwrap();
        model
    }

    pub fn hashed_repr(&self) -> String {
        let json = serde_json::to_string(&self).unwrap();
        let hash = Sha256::digest(json.as_bytes());
        let hash_string = format!("{:x}", hash);
        hash_string
    }

    #[cfg(feature = "data")]
    pub fn with_new_dataset(&mut self, dataset: Dataset) -> Model {
        let mut model = self.clone();
        model.dataset_config = dataset;
        model
    }

    #[cfg(feature = "data")]
    pub fn to_network(&self) -> Network {
        let network_config = self.network.clone().expect("You cannot create a network if it is not configurationified");
        let in_dims = self.dataset_config.in_features_names().len();
        network_config.to_network(in_dims)
    }

    #[cfg(not(feature = "data"))]
    pub fn to_network(&self, in_dims: usize) -> Network {
        let network_config = self.network.clone().expect("You cannot create a network if it is not configurationified");
        network_config.to_network(in_dims)
    }

    #[cfg(feature = "data")]
    pub fn train_epoch(
        &self,
        epoch: usize,
        network: &mut Network,
        train_data: &DataTable,
        id_column: &str,
    ) -> Scalar {
        let (train_x_table, train_y_table) =
            train_data.random_order_in_out(&self.dataset_config.predicted_features_names());

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

    #[cfg(not(feature = "data"))]
    pub fn train_epoch(
        &self,
        epoch: usize,
        network: &mut Network,
        train_x: &Vec<Vec<Scalar>>,
        train_y: &Vec<Vec<Scalar>>,
    ) -> Scalar {
        let train_loss = network.train(
            epoch,
            &train_x,
            &train_y,
            &self.loss.to_loss(),
            self.batch_size.unwrap_or(train_x.len()),
        );

        train_loss
    }

    #[cfg(feature = "data")]
    /// Uses the model's dataset configuration to label the prediction's columns and convert it all to a `DataTable` spreadsheet.
    pub fn preds_to_table(&self, preds: Vec<Vec<Scalar>>) -> DataTable {
        let mut table = DataTable::new_empty();
        let names = self.dataset_config.predicted_features_names();
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
}