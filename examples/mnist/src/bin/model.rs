use neural_networks_rust::model::Model;
use neural_networks_rust::monitor::TasksMonitor;
use neural_networks_rust::preprocessing::attach_ids::AttachIds;
use neural_networks_rust::preprocessing::Pipeline;
use neural_networks_rust::trainers::split::SplitTraining;

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    TasksMonitor::start_monitoring();

    let mut pipeline = Pipeline::basic_single_pass();
    let (dspec, data) = pipeline
        .push(AttachIds::new("id"))
        .load_csv("./dataset/train.csv", &model.dataset)
        .run();

    TasksMonitor::start("modelinit");
    
    let model = model.with_new_dataset(dspec);

    TasksMonitor::end_with_message(format!(
        "Model parameters count: {}",
        model.to_network().get_params().count()
    ));

    let mut training = SplitTraining::new(0.8);
    let (validation_preds, model_eval) = training.run(&model, &data);

    TasksMonitor::stop_monitoring();

    let model_params = training.take_model();
    model_params.to_json(format!("models_stats/{}_params.json", config_name));

    let validation_preds = pipeline.revert(&validation_preds);
    let data = pipeline.revert(&data);
    let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

    data_and_preds.to_csv_file(format!("models_stats/{}.csv", config_name));

    println!("{:#?}", data_and_preds);

    model_eval.to_json_file(format!("models_stats/{}.json", config_name));
}
