use crate::{
    dataset::{Dataset, Feature},
    datatable::DataTable,
};

use super::{CachedConfig, DataTransformation};

pub struct AttachIds(pub String);

impl AttachIds {
    pub fn new(id_column_name: &str) -> Self {
        Self(id_column_name.to_string())
    }
}

impl DataTransformation for AttachIds {
    fn transform(
        &mut self,
        _cached_config: &CachedConfig,
        dataset_config: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let mut feature = Feature::default();
        feature.name = self.0.clone();
        feature.used_in_model = true;
        feature.is_id = true;
        let configuration = dataset_config.with_added_feature(feature);
        let data = data.with_autoincrement_id_column(&self.0.clone());
        (configuration, data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        format!("attach_ids({})", self.0)
    }
}
