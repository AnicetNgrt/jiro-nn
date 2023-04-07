use std::env;

use nn::datatable::DataTable;
use plotly::{Plot, Scatter, ImageFormat, common::{Mode, Title, DashType}, layout::{Axis}, Layout};

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name = &args[1];
    
    let stats = DataTable::from_file(format!("models_stats/{}.csv", model_name));

    let avg_prop_dist = stats.column_to_vecf64("avg price pred prop dist");
    let min_prop_dist = stats.column_to_vecf64("min price pred prop dist");
    // let max_prop_dist = stats.column_to_vecf64("max price pred prop dist");
    let var_prop_dist = stats.column_to_vecf64("var price pred prop dist");

    let mut plot = Plot::new();

    let title = format!("Model {} proportional price prediction distance over epochs", model_name);

    plot.set_layout(
        Layout::new()
            .title(Title::new(title.as_str()))
            .x_axis(Axis::new().title("epoch".into()))
            .y_axis(Axis::new()
                .title("dist(prediction, reality) / reality".into())
                .dtick(0.1)
            ),
    );

    let start = if let Ok(start) = env::var("START_EPOCH") {
        start.parse::<usize>().unwrap()
    } else {
        0
    };

    let min_prop_dist_trace = Scatter::new((start..min_prop_dist.len()).collect(), min_prop_dist)
        .mode(Mode::Lines)
        .name("mininimum")
        .line(plotly::common::Line::new().color("purple"));

    // let max_prop_dist_trace = Scatter::new((0..max_prop_dist.len()).collect(), max_prop_dist)
    //     .mode(Mode::Lines)
    //     .name("maximum")
    //     .line(plotly::common::Line::new().color("pink"));

    let end = avg_prop_dist.len();

    let avg_prop_dist_trace = Scatter::new((start..end).collect(), avg_prop_dist.clone().into_iter().skip(start).collect())
        .mode(Mode::Lines)
        .name("average")
        .line(plotly::common::Line::new().color("blue"));

    let std_prop_dist_trace = Scatter::new((start..end).collect(), var_prop_dist.iter().skip(start).map(|var| var.sqrt()).collect::<Vec<f64>>())
        .mode(Mode::Lines)
        .name("standard deviation")
        .line(plotly::common::Line::new().color("pink").dash(DashType::Dot));

    let low_prop_dist_trace = Scatter::new((start..end).collect(), avg_prop_dist.clone().into_iter().skip(start).zip(var_prop_dist.iter().skip(start)).map(|(avg, var)| avg - var.sqrt()).collect::<Vec<f64>>())
        .mode(Mode::Lines)
        .name("average - standard deviation")
        .line(plotly::common::Line::new().color("red").dash(DashType::Dot));

    let high_prop_dist_trace = Scatter::new((start..end).collect(), avg_prop_dist.into_iter().skip(start).zip(var_prop_dist.iter().skip(start)).map(|(avg, var)| avg + var.sqrt()).collect::<Vec<f64>>())
        .mode(Mode::Lines)
        .name("average + standard deviation")
        .line(plotly::common::Line::new().color("green").dash(DashType::Dot));

    plot.add_trace(min_prop_dist_trace);
    // plot.add_trace(max_prop_dist_trace);
    plot.add_trace(avg_prop_dist_trace);
    plot.add_trace(std_prop_dist_trace);
    plot.add_trace(low_prop_dist_trace);
    plot.add_trace(high_prop_dist_trace);
    
    plot.write_image(format!("visuals/{}_prop_dist.png", model_name), ImageFormat::PNG, 1600, 1200, 1.0);
}