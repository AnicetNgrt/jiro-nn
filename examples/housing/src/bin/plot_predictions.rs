use nn::{datatable::DataTable, charts_utils::Chart};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    let out_data = DataTable::from_file(format!("models_stats/{}.csv", model_name));

    for col in out_data.get_columns_names().iter() {
        Chart::new(format!("Predicted price and price according to {}", col), "price")
            .add_x_axis(col, out_data.column_to_tensor(col))
            .add_discrete_y("predicted price", out_data.column_to_tensor("pred_price"))
            .add_discrete_y("true price", out_data.column_to_tensor("price"))
            .scatter(format!("visuals/{}_{}.png", model_name, col));
    }

    Chart::new("Predicted price and price according to latitude and longitude", "price")
        .add_x_axis("latitude", out_data.column_to_tensor("lat"))
        .add_x_axis("longitude", out_data.column_to_tensor("long"))
        .add_discrete_y("predicted price", out_data.column_to_tensor("pred_price"))
        .add_discrete_y("true price", out_data.column_to_tensor("price"))
        .scatter_3d(format!("visuals/{}_latlong.png", model_name));

    let col1 = "lat";
    let col2 = "long";
    Chart::new(format!("Predicted price and price according to {} and {}", col1, col2), "price")
        .add_x_axis(col1, out_data.column_to_tensor(col1))
        .add_x_axis(col2, out_data.column_to_tensor(col2))
        .add_discrete_y("predicted price", out_data.column_to_tensor("pred_price"))
        .add_discrete_y("true price", out_data.column_to_tensor("price"))
        .scatter_3d(format!("visuals/{}_{}_{}.png", model_name, col1, col2));
}