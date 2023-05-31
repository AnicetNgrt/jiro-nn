use std::collections::HashMap;

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
    linalg::Scalar,
    vec_utils::min_vector,
};

use super::{feature_cached::FeatureExtractorCached, DataTransformation, CachedConfig};

pub struct LogScale10 {
    logged_features: HashMap<String, Scalar>,
}

impl LogScale10 {
    pub fn new() -> Self {
        Self {
            logged_features: HashMap::new(),
        }
    }
}

impl DataTransformation for LogScale10 {
    fn transform(
        &mut self,
        cached_config: &CachedConfig,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut logged_features = HashMap::new();

        for feature in spec.features.iter() {
            if feature.log10 {
                let values = data.column_to_vector(&feature.name);
                let min = min_vector(&values);
                logged_features.insert(feature.name.clone(), min);
            }
        }

        self.logged_features = logged_features.clone();

        let mut extractor = FeatureExtractorCached::new(
            Box::new(move |feature: &Feature| match &feature.with_log10 {
                Some(new_feature) => Some(*new_feature.clone()),
                _ => match &feature.log10 {
                    true => {
                        let mut feature = feature.clone();
                        feature.log10 = false;
                        Some(feature)
                    }
                    _ => None,
                },
            }),
            Box::new(
                move |data: &DataTable, extracted: &Feature, feature: &Feature| {
                    data.map_scalar_column(&feature.name, |x| {
                        let min = logged_features.get(&feature.name).unwrap();
                        if min <= &1.0 {
                            (min.abs() + x + 0.001).log10()
                        } else {
                            x.log10()
                        }
                    })
                    .rename_column(&feature.name, &extracted.name)
                },
            ),
        );

        extractor.transform(cached_config, spec, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        let mut reversed_data = data.clone();

        for (feature, min) in self.logged_features.iter() {
            if reversed_data.has_column(feature) {
                reversed_data = reversed_data.map_scalar_column(feature, |x| {
                    if min <= &1.0 {
                        (10 as Scalar).powf(x) - min.abs() - 0.001
                    } else {
                        (10 as Scalar).powf(x)
                    }
                });
            }
        }

        reversed_data
    }

    fn get_name(&self) -> String {
        "log10".to_string()
    }
}
