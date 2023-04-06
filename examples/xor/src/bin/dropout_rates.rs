use nn::{activation::Activation, layer::{full_layer::FullLayer, dense_layer::DenseLayer, hidden_layer::HiddenLayer, skip_layer::SkipLayer}, network::Network};
use plotters::prelude::*;
use xor::{train_and_test};

const OUT_FILE_NAME: &'static str = "./visuals/dropout_rate.png";

fn avg_error_at_epoch(epochs: usize, dropout_rate: f64, trials: usize) -> Vec<f64> {
    let mut total_errors = Vec::<f64>::from_iter((0..epochs).map(|_| 0.));
    for _ in 0..trials {
        
        let in_to_h1 = FullLayer::<2, 3>::new(
            DenseLayer::new(),
            Activation::Tanh.to_layer(),
        );
        let in_to_h1_config = in_to_h1.get_config();
        let h1_to_out = FullLayer::<3, 1>::new(
            DenseLayer::new(),
            Activation::Tanh.to_layer(),
        );
        let h1_to_out_config = h1_to_out.get_config();
        let mut network = Network::new(
            Box::new(HiddenLayer::new(in_to_h1, SkipLayer::<3>, h1_to_out))
        );
    
        in_to_h1_config.set_dropout_rate(dropout_rate * 0.1);
        h1_to_out_config.set_dropout_rate(dropout_rate);

        let errors = train_and_test(&mut network, epochs, 0.1, move |_, lr| lr).1;
        for (i, e) in errors.into_iter().enumerate() {
            total_errors[i] += e;
        }
    }

    total_errors.into_iter().map(|e| e / trials as f64).collect()
}

fn main() {
    let epochs = 1000;
    let trials = 100;

    let dropout_rate_0 = 0.;
    let errors_0 = avg_error_at_epoch(epochs, dropout_rate_0, trials);

    let dropout_rate_1 = 0.1;
    let errors_1 = avg_error_at_epoch(epochs, dropout_rate_1, trials);

    let dropout_rate_2 = 0.5;
    let errors_2 = avg_error_at_epoch(epochs, dropout_rate_2, trials);


    // println!("[{}]\n testset accuracy: {}", dropout_rate_0, test_set_accuracy(&mut network_0));
    // println!("[{}]\n testset accuracy: {}", dropout_rate_1, test_set_accuracy(&mut network_1));

    let area = BitMapBackend::new(OUT_FILE_NAME, (1024, 760)).into_drawing_area();

    area.fill(&WHITE).unwrap();

    let x_axis = (0f64..(epochs as f64)).step(1f64);
    let xs = 0usize..epochs;

    let mut chart = ChartBuilder::on(&area)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("XOR model predictions", ("sans", 20))
        .build_cartesian_2d(x_axis.clone(), 0f64..0.3f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(20)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .x_desc("epochs")
        .y_labels(20)
        .y_desc(format!("avg error over {} trials", trials))
        .disable_mesh()
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(xs.clone().map(|x| (x as f64, errors_0[x])), &RED)).unwrap()
        .label(format!("dropout_rate = {}", dropout_rate_0))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], &RED));

    chart
        .draw_series(LineSeries::new(xs.clone().map(|x| (x as f64, errors_1[x])), &BLUE)).unwrap()
        .label(format!("dropout_rate = {}", dropout_rate_1))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(xs.clone().map(|x| (x as f64, errors_2[x])), &GREEN)).unwrap()
        .label(format!("dropout_rate = {}", dropout_rate_2))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], &GREEN));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw().unwrap();

    area.present().unwrap();
    println!("Result has been saved to {}", OUT_FILE_NAME);
}