use std::collections::HashSet;

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
};

use super::{feature_cached::FeatureExtractorCached, DataTransformation, CachedConfig};

pub struct Square {
    squared_features: HashSet<String>,
}

impl Square {
    pub fn new() -> Self {
        Self {
            squared_features: HashSet::new(),
        }
    }
}

impl DataTransformation for Square {
    fn transform(
        &mut self,
        cached_config: &CachedConfig,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut squared_features = HashSet::new();

        for feature in spec.features.iter() {
            if feature.squared {
                squared_features.insert(feature.name.clone());
            }
        }

        self.squared_features = squared_features.clone();

        let mut extractor = FeatureExtractorCached::new(
            Box::new(move |feature: &Feature| match &feature.with_squared {
                Some(new_feature) => Some(*new_feature.clone()),
                _ => match &feature.squared {
                    true => {
                        let mut feature = feature.clone();
                        feature.squared = false;
                        Some(feature)
                    }
                    _ => None,
                },
            }),
            Box::new(
                move |data: &DataTable, extracted: &Feature, feature: &Feature| {
                    data.map_scalar_column(&feature.name, |x| x.powi(2))
                        .rename_column(&feature.name, &extracted.name)
                },
            ),
        );

        extractor.transform(cached_config, spec, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        let mut reversed_data = data.clone();

        for feature in self.squared_features.iter() {
            if reversed_data.has_column(feature) {
                reversed_data = reversed_data.map_scalar_column(feature, |x| x.sqrt());
            }
        }

        reversed_data
    }

    fn get_name(&self) -> String {
        "square".to_string()
    }
}
