use gnuplot::{Figure, PlotOption::{Color, Caption, PointSize}, AxesCommon};
use jiro_nn::{datatable::DataTable};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    let out_data = DataTable::from_csv_file(format!("models_stats/{}.csv", model_name));

    for col in out_data.get_columns_names().iter() {
        let mut fg = Figure::new();
        let x = out_data.column_to_vector(col);
        let y1 = out_data.column_to_vector("pred_price");
        let y2 = out_data.column_to_vector("price");
    
        fg.axes2d()
            .set_title(&format!("Predicted price and price according to {}", col), &[])
            .set_x_label(&col.replace("_", " "), &[])
            .set_y_label("price", &[])
            .points(x.clone(), y2.clone(), &[Color("red"), PointSize(0.2), Caption("price")])
            .points(x.clone(), y1.clone(), &[Color("blue"), PointSize(0.2), Caption("predicted price")]);

        fg.save_to_png(format!("visuals/{}_{}.png", model_name, col), 1024, 728).unwrap();
    }

    let mut fg = Figure::new();
    let x = out_data.column_to_vector("lat");
    let z1 = out_data.column_to_vector("pred_price");
    let z2 = out_data.column_to_vector("price");
    let y = out_data.column_to_vector("long");

    fg.axes3d()
        .set_title("Predicted price and price according to latitude and longitude", &[])
        .set_view(45., 15.)
        .set_x_label("latitude", &[])
        .set_y_label("longitude", &[])
        .set_z_label("price", &[])
        .points(x.clone(), y.clone(), z2.clone(), &[Color("red"), PointSize(0.2), Caption("true price")])
        .points(x.clone(), y.clone(), z1.clone(), &[Color("blue"), PointSize(0.2), Caption("predicted price")]);

    fg.save_to_png(format!("visuals/{}_latlong.png", model_name), 1024, 728).unwrap();
}