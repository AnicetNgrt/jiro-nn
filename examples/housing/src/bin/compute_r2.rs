use neural_networks_rust::{datatable::DataTable, vec_utils::r2_score};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    let out_data = DataTable::from_file(format!("models_stats/{}.csv", model_name));

    let y_hat = out_data.column_to_vector("pred_price");
    let y = out_data.column_to_vector("price");

    println!("R²: {}", r2_score(&y_hat, &y));
}