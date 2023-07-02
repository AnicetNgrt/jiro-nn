use neural_networks_rust::model::Model;
use neural_networks_rust::monitor::TM;
use neural_networks_rust::preprocessing::Pipeline;
use neural_networks_rust::trainers::split::SplitTraining;

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    TM::start_monitoring();

    let mut pipeline = Pipeline::basic_single_pass();
    let (dspec, data) = pipeline
        .load_data_and_spec("./dataset/train_cleaned.parquet", &model.dataset)
        .run();

    TM::start("modelinit");
    
    let model = model.with_new_dataset(dspec);

    TM::end_with_message(format!(
        "Model parameters count: {}",
        model.to_network().get_params().count()
    ));

    let mut training = SplitTraining::new(0.8);
    let (preds_and_ids, model_eval) = training.run(&model, &data);

    TM::stop_monitoring();

    let model_params = training.take_model();
    model_params.to_json(format!("models_stats/{}_params.json", config_name));

    let preds_and_ids = pipeline.revert(&preds_and_ids);
    let data = pipeline.revert(&data);

    let values = data.select_columns(&["id", "label"]);
    let values_and_preds = values.inner_join(&preds_and_ids, "id", "id", None);

    data.to_csv_file(format!("models_stats/{}_data_for_values_and_preds.parquet", config_name));
    values_and_preds.to_csv_file(format!("models_stats/{}_values_and_preds.parquet", config_name));

    println!("{:#?}", values_and_preds);

    model_eval.to_json_file(format!("models_stats/{}.json", config_name));
}
