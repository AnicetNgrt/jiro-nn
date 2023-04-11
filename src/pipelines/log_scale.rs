use crate::{dataset::{Dataset, Feature}, datatable::DataTable};

use super::{DataTransformation, feature_cached::FeatureExtractorCached};

pub struct LogScale10;

impl DataTransformation for LogScale10 {
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable) {
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

    fn get_name(&self) -> String {
        "log10".to_string()
    }
}