use crate::{dataset::Dataset, datatable::DataTable};

pub mod feature_cached;
pub mod normalize;
pub mod log_scale;
pub mod to_timestamp;

pub struct Pipeline {
    transformations: Vec<Box<dyn DataTransformation>>,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            transformations: Vec::new(),
        }
    }

    pub fn add(&mut self, transformation: Box<dyn DataTransformation>) -> &mut Pipeline {
        self.transformations.push(transformation);
        self
    }

    pub fn run(&mut self, working_dir: &str, spec: &Dataset) -> (Dataset, DataTable) {
        let data = DataTable::from_file(format!("{}/{}.csv", working_dir, spec.name));
        
        let mut id = spec.name.clone();
        let mut res = (spec.clone(), data.clone());
        for transformation in &mut self.transformations {
            id = format!("{}->{}", id, transformation.get_name());
            res = transformation.transform(&id, working_dir, &res.0, &res.1);
        }
        res
    }
}

pub trait DataTransformation {
    fn get_name(&self) -> String;
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable);
}