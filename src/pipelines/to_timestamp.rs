use polars::export::chrono::{DateTime, NaiveDateTime, Utc};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
};

use super::{feature_cached::FeatureExtractorCached, DataTransformation};

pub struct ToTimestamps;

impl DataTransformation for ToTimestamps {
    fn transform(
        &mut self,
        id: &String,
        working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut feature_cached = FeatureExtractorCached::new(
            Box::new(|feature: &Feature| if feature.date_format.is_some() {
                if let Some(extracted) = &feature.extract_timestamp {
                    Some(*extracted.clone())
                } else {
                    None
                }
            } else {
                None
            }),
            Box::new(|data: &DataTable, extracted: &Feature, feature: &Feature| {
                data.map_str_column_to_f64_column(
                    &feature.name,
                    &extracted.name,
                    |date| {
                        let datetime =
                            NaiveDateTime::parse_from_str(date, "%Y%m%dT%H%M%S").unwrap();
                        let timestamp: DateTime<Utc> = DateTime::from_utc(datetime, Utc);
                        let unix_seconds = timestamp.timestamp();
                        unix_seconds as f64
                    },
                )
            }),
        );

        feature_cached.transform(id, working_dir, spec, data)
    }

    fn get_name(&self) -> String {
        "to_timestamps".to_string()
    }
}
