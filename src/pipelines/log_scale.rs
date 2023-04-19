use std::collections::HashSet;

use crate::{dataset::{Dataset, Feature}, datatable::DataTable};

use super::{DataTransformation, feature_cached::FeatureExtractorCached};

pub struct LogScale10 {
    logged_features: HashSet<String>
}

impl LogScale10 {
    pub fn new() -> Self {
        Self {
            logged_features: HashSet::new(),
        }
    }
}

impl DataTransformation for LogScale10 {
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable) {
        let mut logged_features = HashSet::new();

        for feature in spec.features.iter() {
            if feature.log10 {
                logged_features.insert(feature.name.clone());
            }
        }

        self.logged_features = logged_features.clone();
        
        let mut extractor = FeatureExtractorCached::new(
            Box::new(move |feature: &Feature| {
                match &feature.with_log10 {
                    Some(new_feature) => Some(*new_feature.clone()),
                    _ => match &feature.log10 {
                        true => Some(feature.clone()),
                        _ => None,
                    },
                }
            }),
            Box::new(move |data: &DataTable, extracted: &Feature, feature: &Feature| {
                data.map_f64_column(&feature.name, |x| x.log(10.))
                    .rename_column(&feature.name, &extracted.name)
            }),
        );
        
        extractor.transform(id, working_dir, spec, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        let mut reversed_data = data.clone();

        for feature in self.logged_features.iter() {
            if reversed_data.has_column(feature) {
                reversed_data = reversed_data.map_f64_column(feature, |x| 10f64.powf(x));
            }
        }

        reversed_data
    }

    fn get_name(&self) -> String {
        "log10".to_string()
    }
}