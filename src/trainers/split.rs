use crate::{
    benchmarking::{EpochEvaluation, TrainingEvaluation, ModelEvaluation},
    datatable::DataTable,
    model::Model,
    network::{params::NetworkParams},
    vec_utils::r2_score_matrix, linalg::Scalar, introspect::GI,
};

use super::Trainer;

pub type ReporterClosure = dyn FnMut(usize, EpochEvaluation) -> ();

pub struct SplitTraining {
    pub ratio: Scalar,
    pub real_time_reporter: Option<Box<ReporterClosure>>,
    pub model: Option<NetworkParams>,
    pub all_epochs_validation: bool,
    pub all_epochs_r2: bool,
}

impl SplitTraining {
    pub fn new(ratio: Scalar) -> Self {
        Self {
            ratio,
            real_time_reporter: None,
            all_epochs_validation: false,
            all_epochs_r2: false,
            model: None
        }
    }

    pub fn take_model(&mut self) -> NetworkParams {
        self.model.take().unwrap()
    }

    /// Enables computing the R2 score of the model at the end of each epoch
    /// and reporting it if a real time reporter is attached.
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
    /// - the current epoch
    /// - the evaluation of the current epoch
    /// 
    pub fn attach_real_time_reporter<F>(&mut self, reporter: F) -> &mut Self
    where
        F: FnMut(usize, EpochEvaluation) -> () + 'static,
    {
        self.real_time_reporter = Some(Box::new(reporter));
        self
    }
}

impl Trainer for SplitTraining {
    fn run(&mut self, model: &Model, data: &DataTable) -> (DataTable, ModelEvaluation) {
        GI::start_task("split");

        GI::start_task("init");
        
        let mut validation_preds = DataTable::new_empty();
        let mut model_eval = ModelEvaluation::new_empty();

        let out_features = model.dataset.out_features_names();
        let id_column = model.dataset.get_id_column().unwrap();
        let mut network = model.to_network();

        // Split the data between validation and training
        let (train_table, validation) = data.split_ratio(self.ratio);

        // Shuffle the validation and training set and split it between x and y
        let (validation_x_table, validation_y_table) =
            validation.random_order_in_out(&out_features);

        // Convert the validation set to vectors
        let validation_x = validation_x_table.drop_column(id_column).to_vectors();
        let validation_y = validation_y_table.to_vectors();

        GI::end_task();

        let mut eval = TrainingEvaluation::new_empty();
        let epochs = model.epochs;
        for e in 0..epochs {
            GI::start_task(&format!("epoch[{}/{}]", e, epochs));

            let train_loss = model.train_epoch(e, &mut network, &train_table, id_column);

            let loss_fn = model.loss.to_loss();
            let (preds, loss_avg, loss_std) = if e == model.epochs - 1 || self.all_epochs_validation
            {
                GI::start_task("vloss");
                let vloss = network.predict_evaluate_many(&validation_x, &validation_y, &loss_fn);
                GI::end_task();
                vloss
            } else {
                (vec![], -1.0, -1.0)
            };

            let r2 = if e == model.epochs - 1 || self.all_epochs_r2 {
                GI::start_task("r2");
                let r2 = r2_score_matrix(&validation_y, &preds);
                GI::end_task();
                r2
            } else {
                -1.0
            };

            println!("Epoch {} done", e);
            let epoch_eval = EpochEvaluation::new(train_loss, loss_avg, loss_std, r2);

            // Report the benchmark in real time if expected
            if let Some(reporter) = self.real_time_reporter.as_mut() {
                reporter(e, epoch_eval.clone());
            }

            // Save the predictions if it is the last epoch
            if e == model.epochs - 1 {
                validation_preds = validation_preds.apppend(
                    &DataTable::from_vectors(&out_features, &preds)
                        .add_column_from(&validation_x_table, id_column),
                );
            };

            eval.add_epoch(epoch_eval);

            GI::end_task();
        }

        model_eval.add_fold(eval);

        self.model = Some(network.get_params());

        GI::end_task();

        (validation_preds, model_eval)
    }
}
