use nn::{datatable::DataTable, charts_utils::Chart, vec_utils::map_tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    // let _model = ModelSpec::from_json_file(format!("models/{}.json", model_name));
    let out_data = DataTable::from_file(format!("models_stats/{}.csv", model_name));

    Chart::new("Predicted price and price according to latitude and longitude", "price")
        .add_x_axis("latitude", out_data.column_to_tensor("lat"))
        .add_x_axis("longitude", out_data.column_to_tensor("long"))
        .add_discrete_y("predicted price", out_data.column_to_tensor("pred_price"))
        .add_discrete_y("true price", out_data.column_to_tensor("price"))
        .scatter_3d(format!("visuals/{}_latlong.png", model_name));

    Chart::new("Predicted price and price according to latitude", "price")
        .add_x_axis("latitude", out_data.column_to_tensor("lat"))
        .add_discrete_y("predicted price", out_data.column_to_tensor("pred_price"))
        .add_discrete_y("true price", out_data.column_to_tensor("price"))
        .scatter(format!("visuals/{}_lat.png", model_name));

    Chart::new("Predicted price and price according to longitude", "price")
        .add_x_axis("longitude", out_data.column_to_tensor("long"))
        .add_discrete_y("predicted price", out_data.column_to_tensor("pred_price"))
        .add_discrete_y("true price", out_data.column_to_tensor("price"))
        .scatter(format!("visuals/{}_long.png", model_name));
}