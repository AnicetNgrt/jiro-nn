use std::collections::HashMap;

use nn::{activation::Activation, layer::{full_layer::FullLayer, dense_layer::DenseLayer, skip_layer::SkipLayer, hidden_layer::HiddenLayer}, network::Network};
use plotters::prelude::*;
use xor::train_and_test;

const OUT_FILE_NAME: &'static str = "./visuals/dropout-xor-example-predictions.png";

fn main() {
    let in_to_h1 = FullLayer::<2, 3>::new(
        DenseLayer::new(),
        Activation::Tanh.to_layer(),
    );
    let dr_in = in_to_h1.access_dropout_rate();
    let h1_to_out = FullLayer::<3, 1>::new(
        DenseLayer::new(),
        Activation::Tanh.to_layer(),
    );
    let dr_h1 = h1_to_out.access_dropout_rate();
    let mut network = Network::new(
        Box::new(HiddenLayer::new(in_to_h1, SkipLayer::<3>, h1_to_out))
    );

    {
        let mut dr_in = dr_in.borrow_mut();
        *dr_in = Some(0.01); 
        let mut dr_h1 = dr_h1.borrow_mut();
        *dr_h1 = Some(0.1); 
    }

    let error = train_and_test(&mut network, 100000, 0.1).0;
    println!("[TANH]\n final error: {}", error);

    {
        let mut dr_in = dr_in.borrow_mut();
        *dr_in = None; 
        let mut dr_h1 = dr_h1.borrow_mut();
        *dr_h1 = None; 
    }

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
                format!("{:.2}-{:.2}", x, z), 
                network.predict_iter(vec![x, z])[0]
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
                |x, z| *predictions.get(&format!("{:.2}-{:.2}", x, z)).unwrap(),
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
