use std::collections::{HashMap, HashSet};

use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable, linalg::Scalar,
};

use super::{DataTransformation};

pub struct OneHotEncode;

impl DataTransformation for OneHotEncode {
    fn transform(
        &mut self,
        _: &String,
        _working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut features_values: HashMap<Feature, HashSet<i64>> = HashMap::new();

        data.as_scalar_hashmap().iter().for_each(|(name, values)| {
            let feature = spec.features.iter().find(|f| f.name == *name).unwrap();
            if feature.one_hot_encoded {
                let mut values_set = HashSet::new();
                values.iter().for_each(|v| { values_set.insert(*v as i64); });
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
        data.clone()
    }

    fn get_name(&self) -> String {
        "onehotencode".to_string()
    }
}
