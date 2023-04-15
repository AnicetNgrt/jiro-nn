use std::{fs::File, io::Write};

use serde::{Deserialize, Serialize};

use crate::{charts_utils::Chart};

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    pub folds: Vec<FoldEvaluation>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FoldEvaluation {
    pub epochs: Vec<EpochEvaluation>,
}

#[derive(Clone, Serialize, Deserialize)]
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

    pub fn get_epochs_avg_loss_avg(&self) -> Vec<f64> {
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

    pub fn get_epochs_std_loss_avg(&self) -> Vec<f64> {
        let avg = self.get_epochs_avg_loss_avg();
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

    pub fn get_folds_avg_final_loss_avg(&self) -> f64 {
        let mut sum = 0.0;
        for fold in &self.folds {
            sum += fold.get_final_epoch().test_loss_avg;
        }
        sum / self.folds.len() as f64
    }

    pub fn get_folds_avg_final_loss_std(&self) -> f64 {
        let mut sum = 0.0;
        for fold in &self.folds {
            sum += fold.get_final_epoch().test_loss_std;
        }
        sum / self.folds.len() as f64
    }

    pub fn get_folds_std_final_loss_avg(&self) -> f64 {
        let avg = self.get_folds_avg_final_loss_avg();
        let mut sum = 0.0;
        for fold in &self.folds {
            sum += (fold.get_final_epoch().test_loss_avg - avg).powi(2);
        }
        (sum / self.folds.len() as f64).sqrt()
    }

    pub fn get_folds_std_final_loss_std(&self) -> f64 {
        let avg = self.get_folds_avg_final_loss_std();
        let mut sum = 0.0;
        for fold in &self.folds {
            sum += (fold.get_final_epoch().test_loss_std - avg).powi(2);
        }
        (sum / self.folds.len() as f64).sqrt()
    }

    pub fn from_json_file(path: &str) -> Self {
        let file = File::open(path).unwrap();
        serde_json::from_reader(file).unwrap()
    }

    pub fn to_json_file(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        let json_string = serde_json::to_string_pretty(self).unwrap();
        file.write_all(json_string.as_bytes()).unwrap();
    }

    pub fn get_n_epochs(&self) -> usize {
        self.folds[0].epochs.len()
    }

    pub fn get_n_folds(&self) -> usize {
        self.folds.len()
    }

    //using the chart_utils module
    pub fn plot_epochs_avg_loss(&self, path: &str) {
        Chart::new("Loss over epochs", "loss")
            .add_range_x_axis("epochs", 0., self.get_n_epochs() as f64, 1.)
            .add_y_axis("avg over folds", self.get_epochs_avg_loss_avg())
            .add_y_axis("std over folds", self.get_epochs_std_loss_avg())
            .line_plot(path);
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
