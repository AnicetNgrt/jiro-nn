use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
};

use super::{feature_cached::FeatureExtractorCached, DataTransformation};

pub struct Normalize;

impl DataTransformation for Normalize {
    fn transform(
        &mut self,
        id: &String,
        working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut extractor = FeatureExtractorCached::new(
            Box::new(move |feature: &Feature| match &feature.with_normalized {
                Some(new_feature) => Some(*new_feature.clone()),
                _ => match &feature.normalized {
                    true => Some(feature.clone()),
                    _ => None,
                },
            }),
            Box::new(|data: &DataTable, extracted: &Feature, feature: &Feature| {
                data
                    .normalize_column(&feature.name)
                    .rename_column(&feature.name, &extracted.name)
            }),
        );

        extractor.transform(id, working_dir, spec, data)
    }

    fn get_name(&self) -> String {
        "norm".to_string()
    }
}
