use std::{fs::File, io::Write};

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ModelEvaluation {
    pub folds: Vec<FoldEvaluation>
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct FoldEvaluation {
    pub epochs: Vec<EpochEvaluation>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct EpochEvaluation {
    pub train_loss: f64,
    pub test_loss_avg: f64,
    pub test_loss_std: f64,
}

impl ModelEvaluation {
    pub fn new_empty() -> Self {
        Self {
            folds: vec![],
        }
    }

    pub fn add_fold(&mut self, fold: FoldEvaluation) {
        self.folds.push(fold);
    }

    pub fn epochs_avg_train_loss(&self) -> Vec<f64> {
        let mut avg = vec![0.0; self.folds[0].epochs.len()];
        for fold in &self.folds {
            for (i, epoch) in fold.epochs.iter().enumerate() {
                avg[i] += epoch.train_loss;
            }
        }
        for i in 0..avg.len() {
            avg[i] /= self.folds.len() as f64;
        }
        avg
    }

    pub fn epochs_std_train_loss(&self) -> Vec<f64> {
        let avg = self.epochs_avg_train_loss();
        let mut std = vec![0.0; self.folds[0].epochs.len()];
        for fold in &self.folds {
            for (i, epoch) in fold.epochs.iter().enumerate() {
                std[i] += (epoch.train_loss - avg[i]).powi(2);
            }
        }
        for i in 0..std.len() {
            std[i] = (std[i] / self.folds.len() as f64).sqrt();
        }
        std
    }

    pub fn epochs_avg_test_loss(&self) -> Vec<f64> {
        let mut avg = vec![0.0; self.folds[0].epochs.len()];
        for fold in &self.folds {
            for (i, epoch) in fold.epochs.iter().enumerate() {
                avg[i] += epoch.test_loss_avg;
            }
        }
        for i in 0..avg.len() {
            avg[i] /= self.folds.len() as f64;
        }
        avg
    }

    pub fn epochs_std_test_loss(&self) -> Vec<f64> {
        let avg = self.epochs_avg_test_loss();
        let mut std = vec![0.0; self.folds[0].epochs.len()];
        for fold in &self.folds {
            for (i, epoch) in fold.epochs.iter().enumerate() {
                std[i] += (epoch.test_loss_avg - avg[i]).powi(2);
            }
        }
        for i in 0..std.len() {
            std[i] = (std[i] / self.folds.len() as f64).sqrt();
        }
        std
    }

    pub fn from_json_file<S: AsRef<str>>(path: S) -> Self {
        let file = File::open(path.as_ref()).unwrap();
        serde_json::from_reader(file).unwrap()
    }

    pub fn to_json_file<S: AsRef<str>>(&self, path: S) {
        let mut file = File::create(path.as_ref()).unwrap();
        let json_string = serde_json::to_string_pretty(self).unwrap();
        file.write_all(json_string.as_bytes()).unwrap();
    }

    pub fn get_n_epochs(&self) -> usize {
        self.folds[0].epochs.len()
    }

    pub fn get_n_folds(&self) -> usize {
        self.folds.len()
    }
}

impl FoldEvaluation {
    pub fn new_empty() -> Self {
        Self {
            epochs: vec![],
        }
    }

    pub fn add_epoch(&mut self, epoch: EpochEvaluation) {
        self.epochs.push(epoch);
    }

    pub fn get_final_epoch(&self) -> EpochEvaluation {
        self.epochs[self.epochs.len() - 1].clone()
    }

    pub fn get_final_test_loss_avg(&self) -> f64 {
        self.get_final_epoch().test_loss_avg
    }

    pub fn get_final_test_loss_std(&self) -> f64 {
        self.get_final_epoch().test_loss_std
    }
}

impl EpochEvaluation {
    pub fn new(train_loss: f64, test_loss_avg: f64, test_loss_std: f64) -> Self {
        Self {
            train_loss,
            test_loss_avg,
            test_loss_std,
        }
    }
}