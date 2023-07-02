use std::{
    cell::RefCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::PathBuf,
    rc::Rc
};

use crate::{dataset::Dataset, datatable::DataTable, monitor::TasksMonitor};

use self::{
    extract_months::ExtractMonths, extract_timestamps::ExtractTimestamps,
    filter_outliers::FilterOutliers, log_scale::LogScale10, map::Map, normalize::Normalize,
    one_hot_encode::OneHotEncode, square::Square,
};

pub mod attach_ids;
pub mod extract_months;
pub mod extract_timestamps;
pub mod feature_cached;
pub mod filter_outliers;
pub mod log_scale;
pub mod map;
pub mod normalize;
pub mod one_hot_encode;
pub mod sample;
pub mod square;

pub struct Pipeline {
    transformations: Vec<Rc<RefCell<dyn DataTransformation>>>,
    cached_config: CachedConfig,
    spec: Option<Dataset>,
    data: Option<DataTable>,
}

pub enum CachedConfig {
    NotCached,
    Cached { id: String, working_dir: String },
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            transformations: Vec::new(),
            cached_config: CachedConfig::NotCached,
            spec: None,
            data: None,
        }
    }

    pub fn cached(&mut self, working_dir: &str) -> &mut Self {
        self.cached_config = CachedConfig::Cached {
            id: "".to_string(),
            working_dir: working_dir.to_string(),
        };
        self
    }

    /// Creates a pipeline that does every possible operations once.
    ///
    /// This may not fit your exact usecase, but it's a good starting point.
    ///
    /// The pipeline is:
    /// - One hot encode categorical features if required
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
            .push(OneHotEncode)
            .push(ExtractMonths)
            .push(ExtractTimestamps)
            .push(Map::new())
            .push(LogScale10::new())
            .push(Square::new())
            .push(FilterOutliers)
            .push(Normalize::new());

        pipeline
    }

    pub fn add_shared(
        &mut self,
        transformation: Rc<RefCell<dyn DataTransformation>>,
    ) -> &mut Pipeline {
        self.transformations.push(transformation);
        self
    }

    pub fn push<DT: DataTransformation + 'static>(&mut self, transformation: DT) -> &mut Pipeline {
        self.transformations
            .push(Rc::new(RefCell::new(transformation)));
        self
    }

    pub fn prepend<DT: DataTransformation + 'static>(
        &mut self,
        transformation: DT,
    ) -> &mut Pipeline {
        self.transformations
            .insert(0, Rc::new(RefCell::new(transformation)));
        self
    }

    pub fn load_csv<P>(&mut self, dataset_path: P, spec: &Dataset) -> &mut Self
    where
        P: Into<PathBuf> + ToString,
    {
        TasksMonitor::start("load_csv");

        let pathname = dataset_path.to_string();

        let data =
            DataTable::from_csv_file(dataset_path).select_columns(spec.feature_names().as_slice());

        self.data = Some(data);
        self.spec = Some(spec.clone());

        TasksMonitor::end_with_message(format!("Successfully loaded {:?}: {:?}", pathname, self.data.as_ref().unwrap().describe()));

        self
    }

    pub fn run(&mut self) -> (Dataset, DataTable) {
        TasksMonitor::start("pipeline");

        let data = self.data.clone().unwrap();
        let spec = self.spec.clone().unwrap();

        let mut hasher = DefaultHasher::new();
        spec.hash(&mut hasher);
        let mut id = hasher.finish().to_string();
        let mut res = (spec.clone(), data.clone());

        for transformation in &mut self.transformations {
            let mut transformation = transformation.borrow_mut();

            TasksMonitor::start(&transformation.get_name());

            id = format!("{}-{}", id, transformation.get_name());
            res = transformation.transform(&self.cached_config, &res.0, &res.1);

            TasksMonitor::end();
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

        TasksMonitor::end_with_message(format!("{:?}", data.describe()));

        (spec, data)
    }

    pub fn revert(&mut self, data: &DataTable) -> DataTable {
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
    fn transform(
        &mut self,
        cached_config: &CachedConfig,
        spec: &Dataset,
        data: &DataTable,
    ) -> (Dataset, DataTable);
    fn reverse_columnswise(&mut self, data: &DataTable) -> DataTable;
}
