use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    benchmarking::{EpochEvaluation, ModelEvaluation, TrainingEvaluation},
    datatable::DataTable,
    linalg::{Matrix, MatrixTrait},
    model::Model,
    monitor::TM,
    network::{params::NetworkParams, Network},
    vec_utils::r2_score_vector2,
};

pub type ReporterClosure = dyn FnMut(usize, usize, EpochEvaluation) -> () + Send + Sync;

/// K-Folds trainer
///
/// Trains a model using K-Folds cross validation.
///
/// The model is trained on `k` folds of the data.
///
/// The folds are created by splitting the data into `k` parts.
///
/// The model is trained on `k-1` folds and validated on the remaining fold.
///
/// This process is repeated `k` times, each time using a different fold for validation.
pub struct KFolds {
    pub k: usize,
    pub real_time_reporter: Arc<Option<Mutex<Box<ReporterClosure>>>>,
    pub return_best: bool,
    pub return_avg: bool,
    pub best: Option<NetworkParams>,
    pub avg: Option<NetworkParams>,
    pub all_epochs_validation: bool,
    pub all_epochs_r2: bool,
}

impl KFolds {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            real_time_reporter: Arc::new(None),
            all_epochs_validation: false,
            all_epochs_r2: false,
            return_best: false,
            return_avg: false,
            best: None,
            avg: None,
        }
    }

    /// Enables saving the model of the best fold at the final epoch
    pub fn compute_best_model(&mut self) -> &mut Self {
        self.return_best = true;
        self
    }

    /// Enables computing the average model of all folds at the final epoch
    pub fn compute_avg_model(&mut self) -> &mut Self {
        self.return_avg = true;
        self
    }

    /// Returns the best model of the folds if computed
    pub fn take_best_model(&mut self) -> NetworkParams {
        self.best.take().unwrap()
    }

    /// Returns the average model of the folds if computed
    pub fn take_avg_model(&mut self) -> NetworkParams {
        self.avg.take().unwrap()
    }

    /// Enables computing the R2 score of the model at the end of each epoch
    /// and reporting it if a real time reporter is attached.
    /// 
    /// /!\ Requires `all_epochs_validation` to be enabled.
    ///
    /// /!\ Is time consuming.
    ///
    /// Otherwise computes it only at the end of the final epoch
    pub fn all_epochs_r2(&mut self) -> &mut Self {
        self.all_epochs_r2 = true;
        self
    }

    /// Enables computing the validation score of the model at the end of each epoch
    /// and reporting it if a real time reporter is attached.
    ///
    /// /!\ Is time consuming.
    ///
    /// Otherwise computes it only at the end of the final epoch
    pub fn all_epochs_validation(&mut self) -> &mut Self {
        self.all_epochs_validation = true;
        self
    }

    /// Attaches a real time reporter to the trainer.
    ///
    /// The reporter is a closure that takes as arguments:
    /// - the current fold
    /// - the current epoch
    /// - the evaluation of the current epoch
    ///
    /// The reporter is called on the fold's thread at the end of each epoch.
    /// It must therefore be `Send + Sync + 'static`.
    pub fn attach_real_time_reporter<F>(&mut self, reporter: F) -> &mut Self
    where
        F: FnMut(usize, usize, EpochEvaluation) -> () + Send + Sync + 'static,
    {
        self.real_time_reporter = Arc::new(Some(Mutex::new(Box::new(reporter))));
        self
    }

    fn compute_best(&mut self, model_eval: &ModelEvaluation, trained_models: &Vec<Network>) {
        if self.return_best {
            TM::start("bestfold");
            let mut best_fold = 0;
            let mut best_fold_r2 = 0.0;

            for (i, fold) in model_eval.folds.iter().enumerate() {
                let r2 = fold.get_final_r2();
                if r2 > best_fold_r2 {
                    best_fold = i;
                    best_fold_r2 = r2;
                }
            }

            let best_params = trained_models[best_fold].get_params();
            TM::end_with_message(format!(
                "Best fold: {} with R2: {} and {} parameters",
                best_fold,
                best_fold_r2,
                best_params.count()
            ));
            self.best = Some(best_params);
        }
    }

    fn compute_avg(&mut self, trained_models: &Vec<Network>) {
        if self.return_avg {
            let networks_params = trained_models
                .iter()
                .map(|n| n.get_params())
                .collect::<Vec<_>>();

            let avg_params = NetworkParams::average(&networks_params);
            self.avg = Some(avg_params);
        }
    }

    #[allow(dead_code)]
    fn sequential_k_fold(
        &mut self,
        i: usize,
        model: &Model,
        data: &DataTable,
        preds_and_ids: &Arc<Mutex<DataTable>>,
        model_eval: &Arc<Mutex<ModelEvaluation>>,
        trained_models: &Arc<Mutex<Vec<Network>>>,
        k: usize,
    ) {
        TM::start(format!("{}/{}", i+1, k));
        TM::start("init");
        let predicted_features = model.dataset.predicted_features_names();
        let id_column = model.dataset
            .get_id_column()
            .expect("One feature must be specified as an id in the dataset specification.");
        let mut network = model.to_network();

        // Split the data between validation and training
        let (train_table, validation) = data.split_k_folds(k, i);

        // Shuffle the validation and training set and split it between x and y
        let (validation_x_table, validation_y_table) =
            validation.random_order_in_out(&predicted_features);

        // Convert the validation set to vectors
        let validation_x = validation_x_table.drop_column(id_column).to_vectors();
        let validation_y = validation_y_table.to_vectors();

        let mut fold_eval = TrainingEvaluation::new_empty();
        let epochs = model.epochs;

        TM::end_with_message(format!(
            "Initialized training with {} samples\nInitialized validation with {} samples",
            train_table.num_rows(),
            validation_x_table.num_rows()
        ));

        TM::start("epochs");
        for e in 0..epochs {
            TM::start(&format!("{}/{}", e+1, epochs));
            // Train the model with the k-th folds except the i-th
            let train_loss = model.train_epoch(e, &mut network, &train_table, id_column);

            // Predict all values in the i-th fold
            let loss_fn = model.loss.to_loss();
            let (preds, loss_avg, loss_std) = if e == model.epochs - 1 || self.all_epochs_validation
            {
                let vloss = network.predict_evaluate_many(
                    &validation_x,
                    &validation_y,
                    &loss_fn,
                    model.batch_size.unwrap_or(validation_x.len()),
                );
                vloss
            } else {
                (vec![], -1.0, -1.0)
            };

            // Compute the R2 score	if it is the last epoch
            // (it would be very costly to do it every time)
            let r2 = if e == model.epochs - 1 || self.all_epochs_r2 {
                TM::start("r2");
                let r2 = r2_score_vector2(&validation_y, &preds);
                TM::end_with_message(format!("R2: {}", r2));
                r2
            } else {
                -1.0
            };

            // Build the benchmork of the model for that epoch
            // Useful for plotting the learning curve
            let eval = EpochEvaluation::new(train_loss, loss_avg, loss_std, r2);

            // Report the benchmark in real time if expected
            if let Some(reporter) = self.real_time_reporter.as_ref() {
                reporter.lock().unwrap()(i, e, eval.clone());
            }

            // Save the predictions if it is the last epoch
            if e == model.epochs - 1 {
                let mut vp = preds_and_ids.lock().unwrap();
                *vp = vp.apppend(
                    &DataTable::from_vectors(&predicted_features, &preds)
                        .add_column_from(&validation_x_table, id_column),
                );
            };
            TM::end_with_message(format!("Training Loss: {}\n ", train_loss));

            fold_eval.add_epoch(eval);
        }
        TM::end_with_message(format!("Final performance: {:#?}", fold_eval.get_final_epoch()));

        trained_models.lock().unwrap().push(network);
        model_eval.lock().unwrap().add_fold(fold_eval);
    }

    #[allow(dead_code)]
    fn parallel_k_fold(
        &mut self,
        i: usize,
        model: &Model,
        data: &DataTable,
        preds_and_ids: &Arc<Mutex<DataTable>>,
        model_eval: &Arc<Mutex<ModelEvaluation>>,
        trained_models: &Arc<Mutex<Vec<Network>>>,
        k: usize,
    ) -> thread::JoinHandle<()> {
        TM::start("parr");
        TM::start("init");
        let i = i.clone();
        let model = model.clone();
        let data = data.clone();
        let preds_and_ids = preds_and_ids.clone();
        let model_eval = model_eval.clone();
        let all_epochs_r2 = self.all_epochs_r2;
        let all_epochs_validation = self.all_epochs_validation;
        let reporter = self.real_time_reporter.clone();
        let trained_models = trained_models.clone();

        TM::end_with_message(format!(
            "Will train {} networks with each:\n{} training samples\n{} validation samples",
            k,
            data.num_rows() - data.num_rows() / k,
            data.num_rows() / k
        ));

        let handle = thread::spawn(move || {
            TM::start(&format!("parrfolds[{}]", i));
            TM::start("init");
            let predicted_features = model.dataset.predicted_features_names();
            let id_column = model.dataset.get_id_column().unwrap();
            let mut network = model.to_network();

            // Split the data between validation and training
            let (train_table, validation) = data.split_k_folds(k, i);

            // Shuffle the validation and training set and split it between x and y
            let (validation_x_table, validation_y_table) =
                validation.random_order_in_out(&predicted_features);

            // Convert the validation set to vectors
            let validation_x = validation_x_table.drop_column(id_column).to_vectors();
            let validation_y = validation_y_table.to_vectors();

            TM::end();

            TM::start("epochs");
            let mut fold_eval = TrainingEvaluation::new_empty();
            let epochs = model.epochs;
            for e in 0..epochs {
                TM::start(&format!("{}/{}", e+1, epochs));
                // Train the model with the k-th folds except the i-th
                let train_loss = model.train_epoch(e, &mut network, &train_table, id_column);

                // Predict all values in the i-th fold
                // It is costly and should be done only during the last epoch
                // and made optional for all the others in the future
                let loss_fn = model.loss.to_loss();
                let (preds, loss_avg, loss_std) = if e == model.epochs - 1 || all_epochs_validation
                {
                    let vloss = network.predict_evaluate_many(
                        &validation_x,
                        &validation_y,
                        &loss_fn,
                        model.batch_size.unwrap_or(validation_x.len()),
                    );
                    vloss
                } else {
                    (vec![], -1.0, -1.0)
                };

                // Compute the R2 score	if it is the last epoch
                // (it would be very costly to do it every time)
                let r2 = if e == model.epochs - 1 || (all_epochs_r2 && all_epochs_validation) {
                    TM::start("r2");
                    let r2 = r2_score_vector2(&validation_y, &preds);
                    TM::end_with_message(format!("R2: {}", r2));
                    r2
                } else {
                    -1.0
                };

                // Build the benchmork of the model for that epoch
                // Useful for plotting the learning curve
                let eval = EpochEvaluation::new(train_loss, loss_avg, loss_std, r2);

                // Report the benchmark in real time if expected
                if let Some(reporter) = reporter.as_ref() {
                    reporter.lock().unwrap()(i, e, eval.clone());
                }

                // Save the predictions if it is the last epoch
                if e == model.epochs - 1 {
                    let mut vp = preds_and_ids.lock().unwrap();
                    *vp = vp.apppend(
                        &DataTable::from_vectors(&predicted_features, &preds)
                            .add_column_from(&validation_x_table, id_column),
                    );
                };

                TM::end_with_message(format!("Training Loss: {}\n ", train_loss));

                fold_eval.add_epoch(eval);
            }
            TM::end_with_message(format!(
                "Final performance: {:#?}",
                fold_eval.get_final_epoch()
            ));

            trained_models.lock().unwrap().push(network);
            model_eval.lock().unwrap().add_fold(fold_eval);
        });

        TM::end();

        handle
    }

    /// Runs the k-fold cross validation
    ///
    /// Assumes the data has all the columns corresponding to the model's dataset.
    ///
    /// Assumes both the data and the model's dataset include an id feature.
    /// 
    pub fn run(&mut self, model: &Model, data: &DataTable) -> (DataTable, ModelEvaluation) {
        assert!(!self.all_epochs_r2 || self.all_epochs_validation);

        TM::start("kfolds");

        // Init the data structures for parallel computing
        let preds_and_ids = Arc::new(Mutex::new(DataTable::new_empty()));
        let model_eval = Arc::new(Mutex::new(ModelEvaluation::new_empty()));
        let trained_models = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        let k = self.k;

        TM::start("folds");
        for i in 0..k {
            if Matrix::is_backend_thread_safe() {
                let handle = self.parallel_k_fold(
                    i,
                    model,
                    data,
                    &preds_and_ids,
                    &model_eval,
                    &trained_models,
                    k,
                );
                handles.push(handle);
            } else {
                self.sequential_k_fold(
                    i,
                    model,
                    data,
                    &preds_and_ids,
                    &model_eval,
                    &trained_models,
                    k,
                );
            }
        }
        TM::start("end");

        for handle in handles.into_iter() {
            handle.join().unwrap();
        }

        // Destroy the datastructures for parallel computing
        let preds_and_ids = Arc::try_unwrap(preds_and_ids)
            .unwrap()
            .into_inner()
            .unwrap();
        let model_eval = Arc::try_unwrap(model_eval).unwrap().into_inner().unwrap();
        let trained_models = Arc::try_unwrap(trained_models)
            .unwrap()
            .into_inner()
            .unwrap();

        // Compute the best and average models
        // and store them internally if necessary
        self.compute_best(&model_eval, &trained_models);
        self.compute_avg(&trained_models);

        TM::end();

        (preds_and_ids, model_eval)
    }
}
