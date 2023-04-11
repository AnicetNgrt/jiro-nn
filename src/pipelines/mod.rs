use crate::{dataset::Dataset, datatable::DataTable};

pub mod feature_cached;
pub mod normalize;
pub mod log_scale;
pub mod extract_timestamps;
pub mod extract_months;

pub struct Pipeline {
    transformations: Vec<Box<dyn DataTransformation>>,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            transformations: Vec::new(),
        }
    }

    pub fn add_dyn(&mut self, transformation: Box<dyn DataTransformation>) -> &mut Pipeline {
        self.transformations.push(transformation);
        self
    }

    pub fn add<DT: DataTransformation + 'static>(&mut self, transformation: DT) -> &mut Pipeline {
        self.transformations.push(Box::new(transformation));
        self
    }

    pub fn run(&mut self, working_dir: &str, spec: &Dataset) -> (Dataset, DataTable) {
        let data = DataTable::from_file(format!("{}/{}.csv", working_dir, spec.name))
            .select_columns(spec.feature_names().as_slice());
        
        let mut id = spec.name.clone();
        let mut res = (spec.clone(), data.clone());
        for transformation in &mut self.transformations {
            id = format!("{}-{}", id, transformation.get_name());
            res = transformation.transform(&id, working_dir, &res.0, &res.1);
        }
        let (spec, data) = res;

        let used_features = spec
            .features
            .iter()
            .filter(|f| f.used_in_model)
            .cloned()
            .collect();

        let spec = Dataset {
            features: used_features,
            ..spec
        };

        let data = data.select_columns(spec.feature_names().as_slice());

        (spec, data)
    }
}

pub trait DataTransformation {
    fn get_name(&self) -> String;
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable);
}