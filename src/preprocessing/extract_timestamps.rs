use polars::export::chrono::{DateTime, NaiveDateTime, Utc};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
    linalg::Scalar,
};

use super::{feature_cached::FeatureExtractorCached, CachedConfig, DataTransformation};

pub struct ExtractTimestamps;

impl DataTransformation for ExtractTimestamps {
    fn transform(
        &mut self,
        cached_config: &CachedConfig,
        dataset_config: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let extracted_feature_config = |feature: &Feature| {
            if feature.date_format.is_some() {
                match &feature.with_extracted_timestamp {
                    Some(new_feature) => Some(*new_feature.clone()),
                    _ => match &feature.to_timestamp {
                        true => {
                            let mut f = feature.clone();
                            f.date_format = None;
                            f.to_timestamp = false;
                            Some(f)
                        }
                        _ => None,
                    },
                }
            } else {
                None
            }
        };

        let extract_feature = |data: &DataTable, extracted: &Feature, feature: &Feature| {
            let format = feature.date_format.clone().unwrap();
            data.map_str_column_to_scalar_column(&feature.name, &extracted.name, |date| {
                let datetime = NaiveDateTime::parse_from_str(date, &format).unwrap();
                let timestamp: DateTime<Utc> = DateTime::from_utc(datetime, Utc);
                let unix_seconds = timestamp.timestamp();
                unix_seconds as Scalar
            })
        };

        let mut extractor = FeatureExtractorCached::new(
            Box::new(extracted_feature_config),
            Box::new(extract_feature),
        );

        extractor.transform(cached_config, dataset_config, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        "to_timestamps".to_string()
    }
}
