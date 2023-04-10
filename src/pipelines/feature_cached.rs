use crate::{dataset::{Feature, Dataset}, datatable::DataTable};

use super::DataTransformation;

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
            extract_feature
        }
    }

    fn get_cached_feature_file_name(&self, id: &String, working_dir: &str, feature: &Feature) -> String {
        let file_name = format!("{}/cached/{}_{}.csv", working_dir, id, feature.name);
        file_name
    }
    
    fn get_cached_feature(&self, id: &String, working_dir: &str, feature: &Feature) -> Option<DataTable> {
        let file_name = self.get_cached_feature_file_name(id, working_dir, feature);
        if std::path::Path::new(&file_name).exists() {
            let dataset_table = DataTable::from_file(file_name);
            Some(dataset_table.get_column_as_table(&feature.name))
        } else {
            None
        }
    }
}

impl DataTransformation for FeatureExtractorCached {
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable) {
        let mut new_spec = spec.clone();
        let mut dataset_table = data.clone();
        // create the dataset/cached directory if it does not exist
        std::fs::create_dir_all(format!("{}/cached/", working_dir)).expect("Failed to create cache directory");
     
        for feature in &spec.features {
            if let Some(extracted_feature) = (self.extracted_feature_spec)(feature) {
                if let Some(cached_data) = self.get_cached_feature(&id, working_dir, &extracted_feature) {
                    dataset_table = if extracted_feature.name == feature.name {
                        dataset_table
                            .drop_column(&feature.name)
                            .append_table_as_column(&cached_data)
                    } else {
                        dataset_table.append_table_as_column(&cached_data)
                    };
                } else {
                    dataset_table = (self.extract_feature)(&dataset_table, &extracted_feature, feature);
                    println!("{:#?}", dataset_table);
                    dataset_table
                        .get_column_as_table(&extracted_feature.name)
                        .to_file(self.get_cached_feature_file_name(&id, working_dir, &extracted_feature));
                }
        
                new_spec = if extracted_feature.name == feature.name {
                    spec.with_replaced_feature(&feature.name, extracted_feature)  
                } else {
                    spec.with_added_feature(extracted_feature) 
                };
            }
        }
    
        (new_spec, dataset_table)
    }

    fn get_name(&self) -> String {
        "".to_string()
    }
}