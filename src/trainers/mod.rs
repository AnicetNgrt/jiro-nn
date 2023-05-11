use serde::{Deserialize, Serialize};

use crate::{benchmarking::ModelEvaluation, datatable::DataTable, model::Model, linalg::Scalar};

use self::kfolds::KFolds;
use self::split::SplitTraining;

pub mod kfolds;
pub mod split;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trainers {
    KFolds(usize),
    SplitTraining(Scalar)
}

impl Trainers {
    pub fn maybe_kfold(&self) -> Option<KFolds> {
        match self {
            Trainers::KFolds(k) => Some(KFolds::new(*k)),
            _ => None
        }
    }

    pub fn maybe_split(&self) -> Option<SplitTraining> {
        match self {
            Trainers::SplitTraining(ratio) => Some(SplitTraining::new(*ratio)),
            _ => None
        }
    }
}

pub trait Trainer {
    fn run(&mut self, model: &Model, data: &DataTable) -> (DataTable, ModelEvaluation);
}
