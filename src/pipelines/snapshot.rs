use crate::{datatable::DataTable, dataset::Dataset};

use super::DataTransformation;

pub struct Snapshot(pub Option<DataTable>);

impl Snapshot {
    pub fn new() -> Self {
        Self(None)
    }

    fn snap(&mut self, datatable: DataTable) {
        self.0 = Some(datatable);
    }

    pub fn get(&self) -> Option<&DataTable> {
        self.0.as_ref()
    }
}

impl DataTransformation for Snapshot {
    fn transform(
        &mut self,
        _id: &String,
        _working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        self.snap(data.clone());
        (spec.clone(), data.clone())
    }

    fn get_name(&self) -> String {
        "snap".to_string()
    }
}