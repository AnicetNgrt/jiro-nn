use std::{fs::File};
use std::io::Read;

use serde::{Serialize, Deserialize};
use sha2::{Digest, Sha256};

use crate::{activation::Activation, optimizer::Optimizers, dataset::Dataset};

#[derive(Serialize, Debug, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub epochs: usize,
    pub dropout: Option<Vec<f64>>,
    pub layers: Vec<usize>,
    pub activation: Activation,
    pub optimizer: Optimizers,
    pub batch_size: Option<usize>,
    pub folds: usize,
    pub dataset: Dataset
}

impl ModelSpec {
    pub fn from_json_file(file_name: &str) -> ModelSpec {
        let mut file = File::open(file_name).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        Self::from_json_string(&contents)
    }

    pub fn from_json_string(json: &str) -> ModelSpec {
        let spec: ModelSpec = serde_json::from_str(json).unwrap();
        spec
    }

    pub fn hashed_repr(&self) -> String {
        let json = serde_json::to_string(&self).unwrap();
        let hash = Sha256::digest(json.as_bytes());
        let hash_string = format!("{:x}", hash);
        hash_string
    }
}