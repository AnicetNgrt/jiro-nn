use std::{hash::{Hash, Hasher}, path::Path};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable, monitor::TM,
};

use super::{CachedConfig, DataTransformation};

pub struct FeatureExtractorCached {
    pub extracted_feature_spec: Box<dyn Fn(&Feature) -> Option<Feature>>,
    pub extract_feature: Box<dyn Fn(&DataTable, &Feature, &Feature) -> DataTable>,
}

impl FeatureExtractorCached {
    pub fn new(
        extracted_feature_spec: Box<dyn Fn(&Feature) -> Option<Feature>>,
        extract_feature: Box<dyn Fn(&DataTable, &Feature, &Feature) -> DataTable>,
    ) -> Self {
        Self {
            extracted_feature_spec,
            extract_feature,
        }
    }

    fn get_hashed_id(id: &String) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish().to_string()
    }

    fn get_cached_feature_file_name(
        &self,
        id: &String,
        working_dir: &str,
        feature: &Feature,
    ) -> String {
        let file_name = Path::new(&working_dir)
            .join("cached")
            .join(format!("{}_{}.csv", Self::get_hashed_id(id), feature.name))
            .to_str()
            .unwrap()
            .to_string();
        file_name
    }

    fn get_cached_feature(
        &self,
        id: &String,
        working_dir: &str,
        feature: &Feature,
    ) -> Option<(DataTable, String)> {
        let file_name = self.get_cached_feature_file_name(id, working_dir, feature);
        if std::path::Path::new(&file_name).exists() {
            let dataset_table = DataTable::from_csv_file(file_name.clone());
            Some((dataset_table.get_column_as_table(&feature.name), file_name))
        } else {
            None
        }
    }

    fn transform_no_cache(
        &mut self,
        mut dataset_table: DataTable,
        feature: &Feature,
        extracted_feature: &Feature,
    ) -> DataTable {
        let old_column = dataset_table.get_column_as_table(&feature.name);
        dataset_table = (self.extract_feature)(&dataset_table, &extracted_feature, feature);
        // if the transformation replaced the old column, we need to add it back
        dataset_table =
            if extracted_feature.name != feature.name && !dataset_table.has_column(&feature.name) {
                dataset_table.append_table_as_column(&old_column)
            } else {
                dataset_table
            };
        dataset_table
    }
}

impl DataTransformation for FeatureExtractorCached {
    fn transform(
        &mut self,
        cached_config: &CachedConfig,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut new_spec = spec.clone();
        let mut dataset_table = data.clone();

        if let CachedConfig::Cached { working_dir, .. } = cached_config {
            // create the dataset/cached directory if it does not exist
            std::fs::create_dir_all(format!("{}/cached/", working_dir))
                .expect("Failed to create cache directory");
        }

        for feature in &spec.features {
            if let Some(extracted_feature) = (self.extracted_feature_spec)(feature) {
                if let CachedConfig::Cached { id, working_dir } = cached_config {
                    if let Some((cached_data, cachefile_name)) =
                        self.get_cached_feature(&id, working_dir, &extracted_feature)
                    {
                        TM::start("loadcache");
                        dataset_table = if extracted_feature.name == feature.name {
                            dataset_table
                                .drop_column(&feature.name)
                                .append_table_as_column(&cached_data)
                        } else {
                            dataset_table.append_table_as_column(&cached_data)
                        };
                        TM::end_with_message(format!(
                            "Loaded {} from cache {}",
                            extracted_feature.name,
                            cachefile_name
                        ));
                    } else {
                        dataset_table = self.transform_no_cache(dataset_table, feature, &extracted_feature);
                        dataset_table
                            .get_column_as_table(&extracted_feature.name)
                            .to_csv_file(self.get_cached_feature_file_name(
                                &id,
                                working_dir,
                                &extracted_feature,
                            ));
                    }
                } else {
                    dataset_table = self.transform_no_cache(dataset_table, feature, &extracted_feature);
                }

                new_spec = if extracted_feature.name == feature.name {
                    new_spec.with_replaced_feature(&feature.name, extracted_feature)
                } else {
                    new_spec.with_added_feature(extracted_feature)
                };
            }
        }

        (new_spec, dataset_table)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        "".to_string()
    }
}
