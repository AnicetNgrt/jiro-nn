use std::{sync::{Arc, Mutex}, thread};

use crate::{model::Model, datatable::DataTable, benchmarking::{ModelEvaluation, FoldEvaluation, EpochEvaluation}, linalg::Scalar};

use super::Trainer;

pub struct KFolds {
    pub k: usize,
    pub real_time_reporter: Option<Box<dyn FnMut(KFoldsReport) -> () + Send + Sync>>
}

#[derive(Debug)]
pub struct KFoldsReport {
    pub fold: usize,
    pub epoch: usize,
    pub train_loss: Scalar,
    pub validation_loss: Scalar
}

impl KFolds {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            real_time_reporter: None
        }
    }

    pub fn attach_real_time_reporter<F>(mut self, reporter: F) -> Self
    where
        F: FnMut(KFoldsReport) -> () + Send + Sync + 'static
    {
        self.real_time_reporter = Some(Box::new(reporter));
        self
    }
}

impl Trainer for KFolds {
    /// Runs the k-fold cross validation
    /// 
    /// Assumes the data has all the columns corresponding to the model's dataset.
    /// 
    /// Assumes both the data and the model's dataset include an id feature.
    
    fn run(self, model: &Model, data: &DataTable) -> (DataTable, ModelEvaluation) {
        let validation_preds = Arc::new(Mutex::new(DataTable::new_empty()));
        let model_eval = Arc::new(Mutex::new(ModelEvaluation::new_empty()));
        let reporter = Arc::new(Mutex::new(self.real_time_reporter));
        let mut handles = Vec::new();

        for i in 0..self.k {
            let i = i.clone();
            let model = model.clone();
            let data = data.clone();
            let validation_preds = validation_preds.clone();
            let model_eval = model_eval.clone();
            let reporter = reporter.clone();

            let handle = thread::spawn(move || {
                let out_features = model.dataset.out_features_names();
                let id_column = model.dataset.get_id_column().unwrap();
                let mut network = model.to_network();

                let (train_table, validation) = data.split_k_folds(self.k, i);

                let (validation_x_table, validation_y_table) =
                    validation.random_order_in_out(&out_features);
        
                let validation_x = validation_x_table.drop_column(id_column).to_vectors();
                let validation_y = validation_y_table.to_vectors();

                let mut fold_eval = FoldEvaluation::new_empty();
                let epochs = model.epochs;
                for e in 0..epochs {
                    let train_loss = model.train_epoch(e, &mut network, &train_table, id_column);

                    let loss_fn = model.loss.to_loss();
                    let (preds, loss_avg, loss_std) =
                        network.predict_evaluate_many(&validation_x, &validation_y, &loss_fn);

                    if let Some(reporter) = reporter.lock().unwrap().as_mut() {
                        reporter(KFoldsReport {
                            fold: i,
                            epoch: e,
                            train_loss,
                            validation_loss: loss_avg
                        });
                    }

                    let eval = EpochEvaluation::new(train_loss, loss_avg, loss_std);

                    if e == model.epochs - 1 {
                        let mut vp = validation_preds.lock().unwrap();
                        *vp = vp.apppend(
                            &DataTable::from_vectors(&out_features, &preds)
                                .add_column_from(&validation_x_table, id_column),
                        )
                    };

                    fold_eval.add_epoch(eval);
                }
                model_eval.lock().unwrap().add_fold(fold_eval);
            });

            handles.push(handle);
        }

        for handle in handles.into_iter() {
            handle.join().unwrap();
        }

        let validation_preds = { validation_preds.lock().unwrap().clone() };
        let model_eval = { model_eval.lock().unwrap().clone() };

        (validation_preds, model_eval)
    }
}