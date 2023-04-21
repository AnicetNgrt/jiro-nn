use std::fs::File;
use std::io::Read;

use polars::prelude::NamedFrom;
use polars::series::Series;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::datatable::DataTable;
use crate::layer::dense_layer::DenseLayer;
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
    pub optimizer: Optimizers,
    pub dropout: Option<f64>,
    pub initial_weights_range: Option<(f64, f64)>,
}

impl ModelSpec {
    pub fn from_json_file<S: AsRef<str>>(file_name: S) -> ModelSpec {
        let mut file = File::open(file_name.as_ref()).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        Self::from_json_string(&contents)
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
                    layer_spec.optimizer.clone(),
                    layer_spec.initial_weights_range.unwrap_or((-1.0, 1.0)),
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
}
