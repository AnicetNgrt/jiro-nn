use crate::{dataset::Dataset, datatable::DataTable, vec_utils::vector_quartiles_iqr};

use super::{CachedConfig, DataTransformation};

pub struct FilterOutliers;

impl DataTransformation for FilterOutliers {
    fn transform(
        &mut self,
        _cached_config: &CachedConfig,
        dataset_config: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut data = data.clone();
        for feature in dataset_config.features.iter() {
            if feature.filter_outliers {
                let vals = data.column_to_vector(&feature.name);
                let (_, _, _, min, max) = vector_quartiles_iqr(&vals);
                data = data.filter_by_scalar_column(&feature.name, |x| x >= min && x <= max);
            }
        }
        (dataset_config.clone(), data.clone())
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        "filter_outliers".to_string()
    }
}
