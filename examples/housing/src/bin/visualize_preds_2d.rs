use std::env;

use nn::datatable::DataTable;
use plotly::{Plot, Scatter, ImageFormat, common::{Mode, Title, Marker}, layout::{Axis}, Layout};

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name = &args[1];
    let feature_name = &args[2];

    let mut plot = Plot::new();

    let title = format!("Model {} predicted price & true price according to {}", model_name, feature_name);

    plot.set_layout(
        Layout::new()
            .title(Title::new(title.as_str()))
            .x_axis(Axis::new().title(Title::new(feature_name)))
            .y_axis(Axis::new()
                .title("price".into())
            ),
    );

    let stats = DataTable::from_file(format!("models_preds/{}.csv", model_name))
        .sort_by_column("price");

    let feature: Vec<f64> = stats.column_to_vecf64(&feature_name).iter().map(|f| (f.log10())+3.).collect();
    let true_y: Vec<f64> = stats.column_to_vecf64("price");
    let pred_y = stats.column_to_vecf64("predicted price");

    let true_trace = Scatter::new(feature.clone(), true_y)
        .mode(Mode::Markers)
        .name("true price")
        .marker(Marker::new().color("green").size(5));

    let predicted_trace = Scatter::new(feature.clone(), pred_y)
        .mode(Mode::Markers)
        .name("predicted price")
        .marker(Marker::new().color("purple").size(3));

    plot.add_trace(true_trace);
    plot.add_trace(predicted_trace);
    
    plot.write_image(format!("visuals/{}_price_over_{}.png", model_name, feature_name), ImageFormat::PNG, 1600, 1200, 1.0);
}