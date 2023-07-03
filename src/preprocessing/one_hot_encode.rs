use std::collections::{HashMap, HashSet};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
    linalg::Scalar,
};

use super::{CachedConfig, DataTransformation};

pub struct OneHotEncode;

impl DataTransformation for OneHotEncode {
    fn transform(
        &mut self,
        _cached_config: &CachedConfig,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut features_values: HashMap<Feature, HashSet<i64>> = HashMap::new();

        data.as_scalar_hashmap().iter().for_each(|(name, values)| {
            let feature = spec.features.iter().find(|f| f.name == *name).unwrap();
            if feature.one_hot_encoded {
                let mut values_set = HashSet::new();
                values.iter().for_each(|v| {
                    values_set.insert(*v as i64);
                });
                features_values.insert(feature.clone(), values_set);
            }
        });

        let mut new_spec = spec.clone();

        for (feature, values) in features_values.iter() {
            for value in values.iter() {
                let mut new_feature = feature.clone();
                new_feature.name = format!("{}={}", feature.name, value);
                new_feature.one_hot_encoded = false;
                new_spec = new_spec.with_added_feature(new_feature);
            }
            new_spec = new_spec.without_feature(feature.name.clone());
        }

        let mut new_data = data.clone();
        for (feature, values) in features_values.iter() {
            let column = new_data.column_to_vector(&feature.name);
            new_data = new_data.drop_column(&feature.name);
            let mut rows = vec![vec![0.0 as Scalar; values.len()]; column.len()];
            let mut names = vec![];

            for (i, value) in values.iter().enumerate() {
                for (row, v) in column.iter().enumerate() {
                    if *v as i64 == *value {
                        rows[row][i] = 1.0;
                    } else {
                        rows[row][i] = 0.0;
                    }
                }
                names.push(format!("{}={}", feature.name, value));
            }

            let onehotdata = DataTable::from_vectors(names.as_slice(), &rows);
            new_data = new_data.append_table_as_columns(&onehotdata);
        }

        (new_spec, new_data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        let mut classes: HashMap<String, Vec<String>> = HashMap::new();
        let mut one_hot_encoded: HashMap<String, Vec<(usize, Scalar)>> = HashMap::new();

        // extract any column name like xxxx=yyyy which is a one hot encoded value for yyyy of column xxxx
        data.as_scalar_hashmap().iter().for_each(|(name, values)| {
            let parts: Vec<&str> = name.split("=").collect();
            if parts.len() == 2 {
                let column_name = parts[0].to_string();
                let class = parts[1].to_string();

                if !one_hot_encoded.contains_key(&column_name) {
                    classes.insert(column_name.clone(), vec![class.clone()]);
                    let class_idx = 0;

                    let mut idmaxes = Vec::new();
                    for i in 0..values.len() {
                        idmaxes.push((class_idx, values[i]));
                    }
                    one_hot_encoded.insert(column_name, idmaxes);
                } else {
                    let column_classes = classes.get_mut(&column_name).unwrap();
                    let class_idx = column_classes.len();
                    column_classes.push(class.clone());

                    let idmaxes = one_hot_encoded.get_mut(&column_name).unwrap();
                    for i in 0..values.len() {
                        idmaxes[i].1 = idmaxes[i].1.max(values[i]);
                        if idmaxes[i].1 == values[i] {
                            idmaxes[i].0 = class_idx;
                        }
                    }
                }
            }
        });

        let mut new_data = data.clone();

        // remove old column=class columns for all classes in classes for that column
        for (column_name, classes) in classes.iter() {
            for class in classes.iter() {
                new_data = new_data.drop_column(&format!("{}={}", column_name, class));
            }
        }

        // add new column columns for all columns and put the most likely class in there
        for (column_name, idmaxes) in one_hot_encoded.iter() {
            let mut column_class_values = Vec::new();
            let mut column_confidence_value = Vec::new();
            for (class_idx, confidence) in idmaxes {
                let class_name = classes.get(column_name).unwrap()[*class_idx].clone();
                column_class_values.push(class_name);
                column_confidence_value.push(*confidence);
            }
            new_data = new_data.with_column_string(&column_name, column_class_values.as_slice());
            new_data = new_data.with_column_scalar(&format!("{}.confidence", column_name), column_confidence_value.as_slice());
        }
        
        new_data
    }

    fn get_name(&self) -> String {
        "onehotencode".to_string()
    }
}
