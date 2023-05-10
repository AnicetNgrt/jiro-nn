use std::{fs::File, io::Write, path::{PathBuf}};

use crate::linalg::{Scalar, Matrix, MatrixTrait};

pub struct NetworkParams(pub Vec<Vec<Vec<Scalar>>>);

impl NetworkParams {
    pub fn average(networks: &Vec<Self>) -> Self {
        let mut params = Vec::new();
        
        let layer_count = networks[0].0.len();

        for layer_index in 0..layer_count {
            let mut layer_params = Matrix::from_column_leading_matrix(&networks[0].0[layer_index]);

            for network in networks.iter().skip(1) {
                let other_params = Matrix::from_column_leading_matrix(&network.0[layer_index]);
                layer_params =  layer_params.component_add(&other_params).scalar_div(2.0);
            }

            params.push(layer_params.get_data());
        }
    
        NetworkParams(params)
    }

    pub fn to_json<P: Into<PathBuf>>(&self, path: P) {
        let json = serde_json::json!({ "params": self.0 });
        let mut file = File::create(path.into()).unwrap();
        file.write_all(json.to_string().as_bytes()).unwrap();
    }

    pub fn from_json<P: Into<PathBuf>>(path: P) -> Self {
        let file = File::open(path.into()).unwrap();
        let json: serde_json::Value = serde_json::from_reader(file).unwrap();
        let params = json["params"].as_array().unwrap();

        NetworkParams(
            params
                .iter()
                .map(|x| {
                    x.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| {
                            x.as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_f64().unwrap() as Scalar)
                                .collect()
                        })
                        .collect()
                })
                .collect(),
        )
    }
}