use rust_nn::model::Model;
use rust_nn::pipelines::Pipeline;
use rust_nn::trainers::Trainer;

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    let mut pipeline = Pipeline::basic_single_pass();
    let (updated_dataset_spec, data) = pipeline.run("./dataset", &model.dataset);

    println!("data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);
    
    let kfold = model.trainer.maybe_kfold().expect("Only KFolds trainer is supported");
    let (validation_preds, model_eval) = kfold
        .attach_real_time_reporter(|report| {
            println!("Fold {:3} Epoch {:4} Train avg loss: {:.6} Pred avg loss: {:.6}",
            report.fold, report.epoch, report.train_loss, report.validation_loss);
        })
        .run(&model, &data);

    let validation_preds = pipeline.revert_columnswise(&validation_preds);
    let data = pipeline.revert_columnswise(&data);
    let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

    data_and_preds.to_file(format!("models_stats/{}.csv", config_name));

    println!("{:#?}", data_and_preds);

    model_eval.to_json_file(format!("models_stats/{}.json", config_name));
}
