use std::{fs::File};
use std::io::Read;

use polars::prelude::NamedFrom;
use polars::series::Series;
use serde::{Serialize, Deserialize};
use sha2::{Digest, Sha256};

use crate::datatable::DataTable;
use crate::loss::Losses;
use crate::network::Network;
use crate::nn;
use crate::{activation::Activation, optimizer::Optimizers, dataset::Dataset};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub epochs: usize,
    pub loss: Losses,
    pub dropout: Option<Vec<f64>>,
    pub layers: Vec<usize>,
    pub activation: Activation,
    pub optimizer: Optimizers,
    pub batch_size: Option<usize>,
    pub folds: usize,
    pub dataset: Dataset
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
        let mut layers = vec![self.dataset.in_features_names().len()];
        layers.append(&mut self.layers.clone());
        layers.push(self.dataset.out_features_names().len());
        nn(layers, vec![self.activation], vec![self.optimizer.clone()])
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