use gnuplot::{Figure, AxesCommon, PlotOption::{Color, Caption, FillAlpha, LineStyle}};
use neural_networks_rust::{benchmarking::ModelEvaluation, linalg::Scalar};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_names = &args[1..];
    let mut fg = Figure::new();
    let mut axes = fg.axes2d()
        .set_title("Loss over epochs", &[])
        .set_x_label("epochs", &[])
        .set_y_label("loss", &[])
        .set_y_log(Some(10.));
    
    let colors = &[
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray"
    ];

    for (i, model_name) in model_names.iter().enumerate() {
        let color = colors[i % colors.len()];

        let model_eval =
            ModelEvaluation::from_json_file(format!("models_stats/{}.json", model_name));
    
        let x = (0..model_eval.get_n_epochs()).collect::<Vec<usize>>();
        let y1 = model_eval.epochs_avg_train_loss();
        let y2 = model_eval.epochs_std_train_loss();
        let y1_minus_y2 = y1.iter().zip(y2.iter()).map(|(a, b)| a - b).collect::<Vec<Scalar>>();
        let y1_plus_y2 = y1.iter().zip(y2.iter()).map(|(a, b)| a + b).collect::<Vec<Scalar>>();
    
        let y3 = model_eval.epochs_avg_test_loss();
        let y4 = model_eval.epochs_std_test_loss();
        let y3_minus_y4 = y3.iter().zip(y4.iter()).map(|(a, b)| a - b).collect::<Vec<Scalar>>();
        let y3_plus_y4 = y3.iter().zip(y4.iter()).map(|(a, b)| a + b).collect::<Vec<Scalar>>();
        
        axes = axes
            .lines(x.clone(), y1.clone(), &[Color(color), Caption(&format!("{} train loss", model_name.replace("_", " ")))])
            .lines(x.clone(), y3.clone(), &[Color(color), Caption(&format!("{} test loss", model_name.replace("_", " "))), LineStyle(gnuplot::DashType::Dash)])
            .fill_between(x.clone(), y1_minus_y2, y1_plus_y2, &[Color(color), FillAlpha(0.1)])
            .fill_between(x.clone(), y3_minus_y4, y3_plus_y4, &[Color(color), FillAlpha(0.1)]);
    }
    
    fg.save_to_png(format!("visuals/{}_loss.png", model_names.join("_")), 1024, 728).unwrap();
}
