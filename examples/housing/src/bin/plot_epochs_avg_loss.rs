use nn::benchmarking::ModelEvaluation;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    let model_eval =
        ModelEvaluation::from_json_file(format!("models_stats/{}.json", model_name));
    model_eval.plot_epochs_avg_loss(format!("visuals/{}_epochs_avg_loss.png", model_name));
}
