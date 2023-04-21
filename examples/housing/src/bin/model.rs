use std::sync::{Arc, Mutex};
use std::thread;

use nn::benchmarking::{EpochEvaluation, FoldEvaluation, ModelEvaluation};
use nn::datatable::DataTable;
use nn::model_spec::ModelSpec;
use nn::pipelines::attach_ids::AttachIds;
use nn::pipelines::extract_months::ExtractMonths;
use nn::pipelines::extract_timestamps::ExtractTimestamps;
use nn::pipelines::filter_outliers::FilterOutliers;
use nn::pipelines::log_scale::LogScale10;
use nn::pipelines::normalize::Normalize;
use nn::pipelines::Pipeline;
use nn::pipelines::square::Square;

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let model = ModelSpec::from_json_file(format!("models/{}.json", config_name));
    println!("model: {:#?}", model);

    let mut pipeline = Pipeline::new();
    let (updated_dataset_spec, data) = pipeline
        .add(AttachIds::new("id"))
        .add(ExtractMonths)
        .add(ExtractTimestamps)
        .add(LogScale10::new())
        .add(Square::new())
        .add(FilterOutliers)
        .add(Normalize::new())
        .run("./dataset", &model.dataset);

    //println!("dataset: {:#?}", updated_dataset_spec);
    println!("data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);

    let validation_preds = Arc::new(Mutex::new(DataTable::new_empty()));
    let model_eval = Arc::new(Mutex::new(ModelEvaluation::new_empty()));
    let mut handles = Vec::new();

    for i in 0..model.folds {
        let i = i.clone();
        let model = model.clone();
        let data = data.clone();
        let validation_preds = validation_preds.clone();
        let model_eval = model_eval.clone();

        let handle = thread::spawn(move || {
            let mut network = model.to_network();

            let (train, validation) = data.split_k_folds(model.folds, i);
    
            let (validation_x_table, validation_y_table) =
                validation.random_order_in_out(&model.dataset.out_features_names());
    
            let validation_x = validation_x_table.drop_column("id").to_vectors();
            let validation_y = validation_y_table.to_vectors();
    
            let mut fold_eval = FoldEvaluation::new_empty();
            for e in 0..model.epochs {
                let (train_x_table, train_y_table) =
                    train.random_order_in_out(&model.dataset.out_features_names());
    
                let train_x = train_x_table.drop_column("id").to_vectors();
                let train_y = train_y_table.to_vectors();
    
                let train_loss = network.train(
                    e,
                    &train_x,
                    &train_y,
                    &model.loss.to_loss(),
                    model.batch_size.unwrap_or(train_x.len()),
                );
    
                let (preds, loss_avg, loss_std) =
                    network.predict_evaluate_many(&validation_x, &validation_y, &model.loss.to_loss());
    
                println!(
                    "Fold {:3} Epoch {:4} Train avg loss: {:.6} Pred avg loss: {:.6}",
                    i, e, train_loss, loss_avg
                );
    
                let eval = EpochEvaluation::new(train_loss, loss_avg, loss_std);
    
                if e == model.epochs - 1 {
                    let mut vp = validation_preds.lock().unwrap();
                    *vp = vp.apppend(
                        &DataTable::from_vectors(&model.dataset.out_features_names(), &preds)
                            .add_column_from(&validation_x_table, "id"),
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

    let validation_preds = {
        validation_preds.lock().unwrap().clone()
    };
    let model_eval = {
        model_eval.lock().unwrap().clone()
    };

    let validation_preds = pipeline.revert_columnswise(&validation_preds);
    let data = pipeline.revert_columnswise(&data);
    let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

    data_and_preds.to_file(format!("models_stats/{}.csv", config_name));

    println!("{:#?}", data_and_preds);

    model_eval.to_json_file(format!("models_stats/{}.json", config_name));
}
