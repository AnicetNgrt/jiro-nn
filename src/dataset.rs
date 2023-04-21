use serde::{Serialize, Deserialize};
use serde_aux::field_attributes::bool_true;

#[derive(Default, Serialize, Debug, Deserialize, Clone, Hash)]
pub struct Feature {
    pub name: String,
    #[serde(default)]
    pub out: bool,
    pub date_format: Option<String>,
    #[serde(default)]
    pub to_timestamp: bool,
    #[serde(default)]
    pub extract_month: bool,
    #[serde(default)]
    pub log10: bool,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub filter_outliers: bool,
    #[serde(default)]
    pub squared: bool,
    pub with_extracted_timestamp: Option<Box<Feature>>,
    pub with_extracted_month: Option<Box<Feature>>,
    pub with_log10: Option<Box<Feature>>,
    pub with_normalized: Option<Box<Feature>>,
    pub with_squared: Option<Box<Feature>>,
    #[serde(default="bool_true")]
    pub used_in_model: bool,
    #[serde(default)]
    pub is_id: bool
}

#[derive(Serialize, Debug, Deserialize, Clone, Hash)]
pub struct Dataset {
    pub name: String,
    pub features: Vec<Feature>,
}

impl Dataset {
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

    pub fn feature_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            names.push(feature.name.as_str());
        }
        names
    }

    pub fn in_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if !feature.out && !feature.is_id && !feature.date_format.is_some() {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    pub fn out_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if feature.out {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    pub fn pred_out_features_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for feature in &self.features {
            if feature.out {
                names.push(format!("pred_{}",feature.name));
            }
        }
        names
    }
}