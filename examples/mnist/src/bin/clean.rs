use jiro_nn::datatable::DataTable;
use jiro_nn::monitor::TM;
use jiro_nn::preprocessing::attach_ids::AttachIds;
use jiro_nn::preprocessing::Pipeline;

pub fn main() {
    TM::start_monitoring();

    TM::start("clean");

    if !std::path::Path::new("dataset/train.parquet").exists() {
        TM::start("convert_to_parquet");
        let data_csv = DataTable::from_csv_file("dataset/train.csv");
        data_csv.to_parquet_file("dataset/train.parquet");
        let size_ratio = std::fs::metadata("dataset/train.parquet").unwrap().len() as f64
            / std::fs::metadata("dataset/train.csv").unwrap().len() as f64;

        TM::end_with_message(format!(
            "Converted to parquet with size ratio: {}",
            size_ratio
        ));
    }

    let mut pipeline = Pipeline::new();
    pipeline.push(AttachIds::new("id"));

    pipeline.load_data("dataset/train.parquet", None);
    let (_, preprocessed_data) = pipeline.run();

    TM::start("save");
    preprocessed_data.to_parquet_file("dataset/train_cleaned.parquet");
    TM::end();

    TM::end();

    TM::stop_monitoring();
}
