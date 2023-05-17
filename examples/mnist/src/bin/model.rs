use neural_networks_rust::model::Model;
use neural_networks_rust::pipelines::Pipeline;
use neural_networks_rust::pipelines::attach_ids::AttachIds;
use neural_networks_rust::trainers::Trainer;

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    let mut pipeline = Pipeline::basic_single_pass();
    let (updated_dataset_spec, data) = pipeline
        .push(AttachIds::new("id"))
        .run("./dataset", &model.dataset);

    println!("data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);
    
    let mut training = model.trainer.maybe_split().expect("Only Split trainer is supported");
    let (validation_preds, model_eval) = training
        .attach_real_time_reporter(|epoch, report| {
            println!("Perf report: {:4} {:#?}", epoch, report)
        })
        .run(&model, &data);

    let best_model_params = training.take_model();

    best_model_params.to_json(format!("models_stats/{}_best_params.json", config_name));

    let validation_preds = pipeline.revert_columnswise(&validation_preds);
    let data = pipeline.revert_columnswise(&data);
    let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

    data_and_preds.to_file(format!("models_stats/{}.csv", config_name));

    println!("{:#?}", data_and_preds);

    model_eval.to_json_file(format!("models_weights/{}.json", config_name));
}
