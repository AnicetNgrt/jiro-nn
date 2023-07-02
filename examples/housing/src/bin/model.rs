use neural_networks_rust::model::Model;
use neural_networks_rust::preprocessing::Pipeline;
use neural_networks_rust::preprocessing::attach_ids::AttachIds;
use neural_networks_rust::trainers::kfolds::KFolds;


pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    let mut pipeline = Pipeline::basic_single_pass();
    let (updated_dataset_spec, data) = pipeline
        .push(AttachIds::new("id"))
        .load_csv("./dataset/kc_house_data.csv", &model.dataset)
        .run();

    println!("data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);
    
    let mut kfold = KFolds::new(10);
    let (validation_preds, model_eval) = kfold
        .attach_real_time_reporter(|fold, epoch, report| {
            println!("Perf report: {:2} {:4} {:#?}", fold, epoch, report)
        })
        // .all_epochs_validation()
        // .all_epochs_r2()
        .compute_best_model()
        // .compute_avg_model()
        .run(&model, &data);

    let best_model_params = kfold.take_best_model();
    //let avg_model_params = kfold.take_avg_model();

    //best_model_params.to_json(format!("models_weights/{}_best_params.json", config_name));
    best_model_params.to_binary_compressed(format!("models_weights/{}_best_params.gz", config_name));
    //avg_model_params.to_json(format!("models_stats/{}_avg_params.json", config_name));

    let validation_preds = pipeline.revert(&validation_preds);
    let data = pipeline.revert(&data);
    let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

    data_and_preds.to_csv_file(format!("models_stats/{}.csv", config_name));

    println!("{:#?}", data_and_preds);

    model_eval.to_json_file(format!("models_stats/{}.json", config_name));
}
