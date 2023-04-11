use polars::export::chrono::{DateTime, NaiveDateTime, Utc};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
};

use super::{feature_cached::FeatureExtractorCached, DataTransformation};

pub struct ExtractTimestamps;

impl DataTransformation for ExtractTimestamps {
    fn transform(
        &mut self,
        id: &String,
        working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {

        let extracted_feature_spec = |feature: &Feature| {
            if feature.date_format.is_some() {
                match &feature.with_extracted_timestamp {
                    Some(new_feature) => Some(*new_feature.clone()),
                    _ => match &feature.to_timestamp {
                        true => {
                            let mut f = feature.clone();
                            f.date_format = None;
                            Some(f)
                        },
                        _ => None,
                    },
                }
            } else {
                None
            }
        };

        let extract_feature = |data: &DataTable, extracted: &Feature, feature: &Feature| {
            let format = feature.date_format.clone().unwrap();
            data.map_str_column_to_f64_column(
                &feature.name,
                &extracted.name,
                |date| {
                    let datetime =
                        NaiveDateTime::parse_from_str(date, &format).unwrap();
                    let timestamp: DateTime<Utc> = DateTime::from_utc(datetime, Utc);
                    let unix_seconds = timestamp.timestamp();
                    unix_seconds as f64
                },
            )
        };

        let mut extractor = FeatureExtractorCached::new(
            Box::new(extracted_feature_spec),
            Box::new(extract_feature),
        );

        extractor.transform(id, working_dir, spec, data)
    }

    fn get_name(&self) -> String {
        "to_timestamps".to_string()
    }
}
