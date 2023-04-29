use crate::{
    dataset::{Dataset},
    datatable::DataTable,
};

use super::{DataTransformation};

pub struct Sample {
    pub count: usize,
    pub shuffle: bool,
}

impl Sample {
    pub fn new(count: usize, shuffle: bool) -> Self {
        Self { count, shuffle }
    }
}

impl DataTransformation for Sample {
    fn transform(
        &mut self,
        _id: &String,
        _working_dir: &str,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable) {
        let data = data.sample(Some(self.count), self.shuffle);
        (spec.clone(), data)
    }

    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable {
        data.clone()
    }

    fn get_name(&self) -> String {
        let seed = if self.shuffle {
            rand::random::<u64>()
        } else {
            0
        };

        format!("sample({},{})", self.count, seed)
    }
}
