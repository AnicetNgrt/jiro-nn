use std::{rc::Rc, cell::RefCell, hash::{Hash, Hasher}, collections::hash_map::DefaultHasher};

use crate::{dataset::Dataset, datatable::DataTable};

use self::{attach_ids::AttachIds, extract_months::ExtractMonths, extract_timestamps::ExtractTimestamps, map::Map, log_scale::LogScale10, square::Square, filter_outliers::FilterOutliers, normalize::Normalize};

pub mod feature_cached;
pub mod normalize;
pub mod log_scale;
pub mod extract_timestamps;
pub mod extract_months;
pub mod attach_ids;
pub mod square;
pub mod filter_outliers;
pub mod map;

pub struct Pipeline {
    transformations: Vec<Rc<RefCell<dyn DataTransformation>>>,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            transformations: Vec::new(),
        }
    }

    /// Creates a pipeline that does every possible operations once.
    ///
    /// This may not fit your exact usecase, but it's a good starting point.
    /// 
    /// The pipeline is:
    /// - Extract months if required
    /// - Extract timestamps if required
    /// - Map values if required
    /// - Log scale if required
    /// - Square values if required
    /// - Filter outliers if required
    /// - Normalize values if required
    /// 
    pub fn basic_single_pass() -> Pipeline {
        let mut pipeline = Pipeline::new();
        pipeline
            .add(AttachIds::new("id"))
            .add(ExtractMonths)
            .add(ExtractTimestamps)
            .add(Map::new())
            .add(LogScale10::new())
            .add(Square::new())
            .add(FilterOutliers)
            .add(Normalize::new());
        
        pipeline
    }

    pub fn add_shared(&mut self, transformation: Rc<RefCell<dyn DataTransformation>>) -> &mut Pipeline {
        self.transformations.push(transformation);
        self
    }

    pub fn add<DT: DataTransformation + 'static>(&mut self, transformation: DT) -> &mut Pipeline {
        self.transformations.push(Rc::new(RefCell::new(transformation)));
        self
    }

    pub fn run(&mut self, working_dir: &str, spec: &Dataset) -> (Dataset, DataTable) {
        let data = DataTable::from_file(format!("{}/{}.csv", working_dir, spec.name))
            .select_columns(spec.feature_names().as_slice());
        
        let mut hasher = DefaultHasher::new();
        spec.hash(&mut hasher);
        let mut id = hasher.finish().to_string();
        let mut res = (spec.clone(), data.clone());

        for transformation in &mut self.transformations {
            let mut transformation = transformation.borrow_mut();
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

    pub fn revert_columnswise(&mut self, data: &DataTable) -> DataTable {
        let mut res = data.clone();

        for transformation in &mut self.transformations.iter().rev() {
            let mut transformation = transformation.borrow_mut();
            res = transformation.reverse_columnswise(&res);
        }
        res
    }
}

pub trait DataTransformation {
    fn get_name(&self) -> String;
    fn transform(&mut self, id: &String, working_dir: &str, spec: &Dataset, data: &DataTable) -> (Dataset, DataTable);
    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable;
}