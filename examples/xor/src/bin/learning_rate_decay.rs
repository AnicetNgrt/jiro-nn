use nn::{activation::Activation, nn};
use plotters::prelude::*;
use xor::{train_and_test};

const OUT_FILE_NAME: &'static str = "./visuals/learning_rate_decay.png";

fn avg_error_at_epoch(epochs: usize, decay: f64, trials: usize) -> Vec<f64> {
    let mut total_errors = Vec::<f64>::from_iter((0..epochs).map(|_| 0.));
    for _ in 0..trials {
        let mut network = nn(vec![Activation::Tanh], vec![2, 3, 1]);
        let errors = train_and_test(&mut network, epochs, 0.1, move |e, lr| lr / (1. + decay * (e as f64))).1;
        for (i, e) in errors.into_iter().enumerate() {
            total_errors[i] += e;
        }
    }

    total_errors.into_iter().map(|e| e / trials as f64).collect()
}

fn main() {
    let epochs = 1000;
    let trials = 100;

    let decay_0 = 0.;
    let errors_0 = avg_error_at_epoch(epochs, decay_0, trials);

    let decay_1 = 0.1;
    let errors_1 = avg_error_at_epoch(epochs, decay_1, trials);

    let decay_2 = 0.01;
    let errors_2 = avg_error_at_epoch(epochs, decay_2, trials);

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
        .label(format!("Decay = {}", decay_0))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], &RED));

    chart
        .draw_series(LineSeries::new(xs.clone().map(|x| (x as f64, errors_1[x])), &BLUE)).unwrap()
        .label(format!("Decay = {}", decay_1))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(xs.clone().map(|x| (x as f64, errors_2[x])), &GREEN)).unwrap()
        .label(format!("Decay = {}", decay_2))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], &GREEN));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw().unwrap();

    area.present().unwrap();
    println!("Result has been saved to {}", OUT_FILE_NAME);
}
