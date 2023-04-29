use serde::{Serialize, Deserialize};

use crate::{model::Model, datatable::DataTable, benchmarking::ModelEvaluation};

use self::kfolds::KFolds;

pub mod kfolds;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trainers {
    KFolds(usize),
}

impl Trainers {
    pub fn maybe_kfold(&self) -> Option<KFolds> {
        match self {
            Trainers::KFolds(k) => Some(KFolds::new(*k)),
        }
    }
}

pub trait Trainer {
    fn run(&mut self, model: &Model, data: &DataTable) -> (DataTable, ModelEvaluation);
}