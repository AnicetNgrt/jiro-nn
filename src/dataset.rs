use serde::{Serialize, Deserialize};
use sha2::{Digest, Sha256};

#[derive(Default, Serialize, Debug, Deserialize, Clone, Hash)]
pub struct Feature {
    pub name: String,
    pub date_format: Option<String>,
    pub extract_timestamp: Option<Box<Feature>>,
    pub extract_month: Option<Box<Feature>>,
    pub extract_log10: Option<Box<Feature>>,
    pub extract_normalized: Option<Box<Feature>>,
    pub extract_squared: Option<Box<Feature>>,
    #[serde(default)]
    pub out: bool,
}

#[derive(Serialize, Debug, Deserialize, Hash, Clone)]
pub struct Dataset {
    pub name: String,
    pub features: Vec<Feature>,
}

impl Dataset {
    pub fn hashed_repr(&self) -> String {
        let json = serde_json::to_string(&self).unwrap();
        let hash = Sha256::digest(json.as_bytes());
        let hash_string = format!("{:x}", hash);
        hash_string
    }

    pub fn with_added_feature(&self, feature: Feature) -> Self {
        let mut features = self.features.clone();
        features.push(feature);
        Self {
            name: self.name.clone(),
            features
        }
    }

    pub fn with_replaced_feature(&self, old_feature_name: &str, feature: Feature) -> Self {
        let mut features = self.features.clone();
        let index = features.iter().position(|f| f.name == old_feature_name).unwrap();
        features[index] = feature;
        Self {
            name: self.name.clone(),
            features
        }
    }
}