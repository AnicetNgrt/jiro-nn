use std::{fs::File, io::{Write, Read}, path::PathBuf};

use flate2::{write::GzEncoder, Compression, read::GzDecoder};
use serde::{Deserialize, Serialize};

use crate::linalg::{Matrix, MatrixTrait, Scalar};

#[derive(Serialize, Deserialize)]
pub struct NetworkParams(pub Vec<Vec<Vec<Scalar>>>);

impl NetworkParams {
    pub fn average(networks: &Vec<Self>) -> Self {
        let mut params = Vec::new();

        let layer_count = networks[0].0.len();

        for layer_index in 0..layer_count {
            let mut layer_params = Matrix::from_column_leading_matrix(&networks[0].0[layer_index]);

            for network in networks.iter().skip(1) {
                let other_params = Matrix::from_column_leading_matrix(&network.0[layer_index]);
                layer_params = layer_params.component_add(&other_params).scalar_div(2.0);
            }

            params.push(layer_params.get_data());
        }

        NetworkParams(params)
    }

    pub fn to_json<P: Into<PathBuf>>(&self, path: P) {
        let json = serde_json::to_value(self).unwrap();
        let mut file = File::create(path.into()).unwrap();
        file.write_all(json.to_string().as_bytes()).unwrap();
    }

    pub fn from_json<P: Into<PathBuf>>(path: P) -> Self {
        let file = File::open(path.into()).unwrap();
        let params: serde_json::Value = serde_json::from_reader(file).unwrap();
        serde_json::from_value(params).unwrap()
    }

    pub fn to_binary_compressed<P: Into<PathBuf>>(&self, path: P) {
        let result = bincode::serialize(self).unwrap();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(result.as_slice()).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut file = File::create(path.into()).unwrap();
        file.write_all(&compressed).unwrap();
    }

    pub fn from_binary_compressed<P: Into<PathBuf>>(path: P) -> Self {
        let file = File::open(path.into()).unwrap();
        let mut decoder = GzDecoder::new(file);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).unwrap();
        bincode::deserialize(buffer.as_slice()).unwrap()
    }
}
