use polars::export::chrono::{DateTime, NaiveDateTime, Utc, Datelike};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
};

use super::{feature_cached::FeatureExtractorCached, DataTransformation};

pub struct ExtractMonths;

impl DataTransformation for ExtractMonths {
    fn transform(
        &mut self,
        id: &String,
        working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {

        let extracted_feature_spec = |feature: &Feature| {
            if feature.date_format.is_some() {
                match &feature.with_extracted_month {
                    Some(new_feature) => Some(*new_feature.clone()),
                    None => match &feature.extract_month {
                        true => {
                            let mut f = feature.clone();
                            f.date_format = None;
                            f.extract_month = false;
                            Some(f)
                        },
                        false => None,
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
                    timestamp.month() as f64
                },
            )
        };

        let mut extractor = FeatureExtractorCached::new(
            Box::new(extracted_feature_spec),
            Box::new(extract_feature),
        );

        extractor.transform(id, working_dir, spec, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        "extract_months".to_string()
    }
}
