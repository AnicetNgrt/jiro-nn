use std::env;

use nn::datatable::DataTable;
use plotly::{Plot, Scatter, ImageFormat, common::{Mode, Title, Line}, layout::{Axis}, Layout};

fn avg_in_subds(min: f64, size: f64, count: usize, xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    let mut subds_ys = vec![vec![]; count];

    for (i, x) in xs.iter().enumerate() {
        let subd_i = ((x - min).div_euclid(size) as usize).min(count-1);
        //println!("{} {}", count, subd_i);
        subds_ys[subd_i].push(ys[i]); 
    }

    subds_ys.into_iter().map(|ys: Vec<f64>| {
        let sum: f64 = ys.iter().sum();
        sum / (ys.len() as f64)
    }).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name = &args[1];
    let feature_name = &args[2];
    let subd_count = &args[3].parse::<usize>().unwrap();

    let mut plot = Plot::new();

    let log_scale = match std::env::var("LOG_SCALE") {
        Ok(val) => val == "true",
        Err(_) => false,
    };

    let title = format!("Model {} average predicted price & true price according to {}", model_name, feature_name);

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

    let feature: Vec<f64> = stats.column_to_vecf64(&feature_name).iter().map(|f| if log_scale { f.log10() } else { *f }).collect();
    println!("{:?}", feature);
    
    let min_feature = feature.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
    let max_feature = feature.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    let range = max_feature - min_feature;
    let subd_size = range / (*subd_count as f64);

    println!("max {} min {}", max_feature, min_feature);
    println!("range: {}", range);
    println!("subd_size: {}", subd_size);

    let subds: Vec<_> = (0..*subd_count).map(|i| min_feature + subd_size * i as f64).collect();
    
    let true_y: Vec<f64> = stats.column_to_vecf64("price");
    let pred_y = stats.column_to_vecf64("predicted price");

    println!("{:?}", subds);

    let true_trace = Scatter::new(subds.clone(), avg_in_subds(*min_feature, subd_size, *subd_count, feature.clone(), true_y))
        .mode(Mode::Lines)
        .name("true price")
        .line(Line::new().color("green"));

    let predicted_trace = Scatter::new(subds.clone(), avg_in_subds(*min_feature, subd_size, *subd_count, feature, pred_y))
        .mode(Mode::Lines)
        .name("predicted price")
        .line(Line::new().color("purple"));

    plot.add_trace(true_trace);
    plot.add_trace(predicted_trace);

    plot.write_image(format!("visuals/{}_price_avg_over_{}.png", model_name, feature_name), ImageFormat::PNG, 1600, 1200, 1.0);
}