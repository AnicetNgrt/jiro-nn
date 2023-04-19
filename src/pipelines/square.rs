use std::collections::HashSet;

use crate::{dataset::{Dataset, Feature}, datatable::DataTable};

use super::{DataTransformation, feature_cached::FeatureExtractorCached};

pub struct Square {
    squared_features: HashSet<String>
}

impl Square {
    pub fn new() -> Self {
        Self {
            squared_features: HashSet::new(),
        }
    }
}

impl DataTransformation for Square {
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable) {
        let mut squared_features = HashSet::new();

        for feature in spec.features.iter() {
            if feature.squared {
                squared_features.insert(feature.name.clone());
            }
        }

        self.squared_features = squared_features.clone();
        
        let mut extractor = FeatureExtractorCached::new(
            Box::new(move |feature: &Feature| {
                match &feature.with_squared {
                    Some(new_feature) => Some(*new_feature.clone()),
                    _ => match &feature.squared {
                        true => Some(feature.clone()),
                        _ => None,
                    },
                }
            }),
            Box::new(move |data: &DataTable, extracted: &Feature, feature: &Feature| {
                data.map_f64_column(&feature.name, |x| x.powi(2))
                    .rename_column(&feature.name, &extracted.name)
            }),
        );
        
        extractor.transform(id, working_dir, spec, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        let mut reversed_data = data.clone();

        for feature in self.squared_features.iter() {
            if reversed_data.has_column(feature) {
                reversed_data = reversed_data.map_f64_column(feature, |x| x.sqrt());
            }
        }

        reversed_data
    }

    fn get_name(&self) -> String {
        "square".to_string()
    }
}