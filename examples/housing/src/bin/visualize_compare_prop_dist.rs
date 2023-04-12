fn main() {
    
}

// use housing::colors;
// use std::env;

// use nn::datatable::DataTable;
// use plotly::{
//     common::{Mode, Title},
//     layout::Axis,
//     ImageFormat, Layout, Plot, Scatter,
// };

// fn main() {
//     let args: Vec<String> = env::args().collect();

//     let mut plot = Plot::new();

//     let title = "Comparative average proportional price prediction distance over epochs";

//     // get dtick from args env var DTICK if present
//     let dtick = if let Ok(dtick) = env::var("DTICK") {
//         dtick.parse::<f64>().unwrap()
//     } else {
//         0.1
//     };

//     plot.set_layout(
//         Layout::new()
//             .title(Title::new(title))
//             .x_axis(Axis::new().title("epoch".into()))
//             .y_axis(
//                 Axis::new()
//                     .title("dist(prediction, reality) / reality".into())
//                     .dtick(dtick),
//             ),
//     );

//     // generate colors for each model with good contrast between them
//     let colors = colors(args.len() - 1);

//     for i in 1..args.len() {
//         let model_name = &args[i];
//         let stats = DataTable::from_file(format!("models_stats/{}.csv", model_name));
//         let avg_prop_dist = stats.column_to_vecf64("avg price pred prop dist");
//         // let var_prop_dist = stats.column_to_vecf64("var price pred prop dist");
//         // let std_prop_dist = var_prop_dist.iter().map(|var| var.sqrt()).collect::<Vec<f64>>();

//         let avg_prop_dist_trace =
//             Scatter::new((0..avg_prop_dist.len()).collect(), avg_prop_dist.clone())
//                 .mode(Mode::Lines)
//                 .name(format!("{} average", model_name))
//                 .line(plotly::common::Line::new().color(colors[i-1]));

//         plot.add_trace(avg_prop_dist_trace);
        
//         // let std_prop_dist_trace =
//         // Scatter::new((0..avg_prop_dist.len()).collect(), std_prop_dist)
//         //     .mode(Mode::Lines)
//         //     .name(format!("{} standard deviation", model_name))
//         //     .line(plotly::common::Line::new().color(colors[i-1]).dash(DashType::Dot));

//         // plot.add_trace(std_prop_dist_trace);
//     }

//     let concatenated_names = args[1..].join("_");
//     plot.write_image(
//         format!("visuals/{}_comp_prop_dist.png", concatenated_names),
//         ImageFormat::PNG,
//         1600,
//         1200,
//         1.0,
//     );
// }
