use std::collections::HashMap;

use nn::{activation::Activation, nn_h1};
use plotters::prelude::*;
use xor::{train_and_test, test_set_accuracy};

const OUT_FILE_NAME: &'static str = "./visuals/xor-example-predictions.png";

fn main() {
    let mut network = nn_h1::<2, 3, 1>(vec![Activation::Tanh]);

    let decay = 0.001;
    let error = train_and_test(&mut network, 10000, 0.1, move |e, lr| lr / (1. + decay * (e as f64))).0;
    println!("[TANH]\n final error: {}", error);
    println!("[TANH]\n testset accuracy: {}", test_set_accuracy(&mut network));

    let area = BitMapBackend::new(OUT_FILE_NAME, (1024, 760)).into_drawing_area();

    area.fill(&WHITE).unwrap();

    let step = 0.01f64;
    let in_min = 0f64;
    let in_max = 1f64;
    let out_min = 0f64;
    let out_max = 1f64;

    let x_axis = (in_min..in_max).step(step);
    let xs = ((in_min as i64)..((in_max/step).round() as i64)).map(|x| x as f64 * step);
    let z_axis = (in_min..in_max).step(step);
    let zs = ((in_min as i64)..((in_max/step).round() as i64)).map(|x| x as f64 * step);

    let mut predictions: HashMap<String, f64> = HashMap::new();
    for x in xs.clone() {
        for z in zs.clone() {
            predictions.insert(
                format!("{}-{}", x, z), 
                network.predict(vec![x, z])[0]
            );
        }
    }

    let mut chart = ChartBuilder::on(&area)
        .caption("XOR model predictions", ("sans", 20))
        .build_cartesian_3d(x_axis.clone(), out_min..out_max, z_axis.clone())
        .unwrap();

    chart.with_projection(|mut pb| {
        pb.yaw = 0.5;
        pb.scale = 0.9;
        pb.pitch = 0.3;
        pb.into_matrix()
    });
    
    chart
        .configure_axes()
        .light_grid_style(BLACK.mix(0.15))
        .max_light_lines(1)
        .draw().unwrap();

    chart
        .draw_series(
            SurfaceSeries::xoz(xs, zs,
                |x, z| *predictions.get(&format!("{}-{}", x, z)).unwrap(),
            )
            .style(BLUE.mix(0.2).filled()),
        ).unwrap()
        .label("Surface")
        .legend(|(x, y)| Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], BLUE.mix(0.5).filled()));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw().unwrap();

    area.present().unwrap();
    println!("Result has been saved to {}", OUT_FILE_NAME);
}
