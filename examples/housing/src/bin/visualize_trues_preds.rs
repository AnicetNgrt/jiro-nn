fn main() {
    
}

// use std::env;

// use housing::colors;
// use nn::datatable::DataTable;
// use plotly::{Plot, Scatter, ImageFormat, common::{Mode, Title, Marker}, layout::{Axis}, Layout};

// fn main() {
//     let args: Vec<String> = env::args().collect();

//     let mut plot = Plot::new();

//     let title = format!("Comparative predicted price according to true_price");

//     plot.set_layout(
//         Layout::new()
//             .title(Title::new(title.as_str()))
//             .x_axis(Axis::new().title("true price".into()).dtick(10000.))
//             .y_axis(Axis::new()
//                 .title("predicted price".into())
//                 .dtick(0.1)
//             ),
//     );

//     let colors = colors(args.len() - 1);

//     for i in 1..args.len() {
//         let model_name = &args[i];
    
//         let stats = DataTable::from_file(format!("models_preds/{}.csv", model_name))
//             .sort_by_column("price");

//         let true_y: Vec<f64> = stats.column_to_vecf64("price").into_iter().map_while(|v| if v < 500000. { Some(v) } else { None }).collect();
//         let pred_y = stats.column_to_vecf64("predicted price");

//         let trace = Scatter::new(true_y.clone(), true_y.iter().zip(pred_y.iter()).map(|(t, p)| (p - t).abs()/t).collect::<Vec<f64>>())
//             .mode(Mode::Markers)
//             .name(model_name)
//             .marker(Marker::new().color(colors[i-1]).size(3));

//         plot.add_trace(trace);

//     }
    
//     let concatenated_names = args[1..].join("_");
//     plot.write_image(format!("visuals/{}_trues_preds.png", concatenated_names), ImageFormat::PNG, 1600, 1200, 1.0);
// }