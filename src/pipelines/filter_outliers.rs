use crate::{
    dataset::{Dataset},
    datatable::DataTable, vec_utils::tensor_boxplot,
};

use super::{DataTransformation};

pub struct FilterOutliers;

impl DataTransformation for FilterOutliers {
    fn transform(
        &mut self,
        _id: &String,
        _working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut data = data.clone();
        for feature in spec.features.iter() {
            if feature.filter_outliers {
                println!("FO: {}", feature.name);
                let vals = data.column_to_tensor(&feature.name);
                let (_, _, _, min, max) = tensor_boxplot(&vals);
                data = data.filter_by_f64_column(&feature.name, |x| x >= min && x <= max);
            }
        }
        (spec.clone(), data.clone())
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        "filter_outliers".to_string()
    }
}
