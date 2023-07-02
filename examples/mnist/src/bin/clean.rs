use neural_networks_rust::monitor::TM;
use neural_networks_rust::preprocessing::attach_ids::AttachIds;
use neural_networks_rust::preprocessing::Pipeline;

pub fn main() {
    TM::start_monitoring();

    let mut pipeline = Pipeline::new();
    pipeline.push(AttachIds::new("id"));

    pipeline.load_data("dataset/train.csv");
    let (_, preprocessed_data) = pipeline.run();

    TM::start("save_csv");
    preprocessed_data.to_csv_file(format!("dataset/train_cleaned.csv"));
    TM::end();

    TM::stop_monitoring();
}
