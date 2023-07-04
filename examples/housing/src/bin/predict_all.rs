use jiro_nn::{
    datatable::DataTable,
    model::Model,
    network::params::NetworkParams,
    preprocessing::{attach_ids::AttachIds, Pipeline},
    vec_utils::r2_score_matrix,
};

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];
    let weights_file = &args[2];
    let out_name = &args[3];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    let mut pipeline = Pipeline::basic_single_pass();
    let (updated_dataset_spec, data) = pipeline
        .push(AttachIds::new("id"))
        .load_data_and_spec("./dataset/kc_house_data.csv", &model.dataset)
        .run();

    println!("data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);
    let predicted_features = model.dataset.predicted_features_names();

    let (x_table, y_table) = data.random_order_in_out(&predicted_features);

    let x = x_table.drop_column("id").to_vectors();
    let y = y_table.to_vectors();

    let weights = NetworkParams::from_json(format!("models_weights/{}.json", weights_file));
    let mut network = model.to_network();
    network.load_params(&weights);

    let (preds, avg_loss, std_loss) =
        network.predict_evaluate_many(&x, &y, &model.loss.to_loss(), 1);

    println!("avg_loss: {:#?}", avg_loss);
    println!("std_loss: {:#?}", std_loss);

    let r2 = r2_score_matrix(&y, &preds);

    println!("r2: {:#?}", r2);

    let preds_and_ids =
        DataTable::from_vectors(&predicted_features, &preds).add_column_from(&x_table, "id");

    let preds_and_ids = pipeline.revert(&preds_and_ids);
    let data = pipeline.revert(&data);
    let data_and_preds = data.inner_join(&preds_and_ids, "id", "id", Some("pred"));

    data_and_preds.to_csv_file(format!("models_stats/{}.csv", out_name));
}
