use gnuplot::*;
use neural_networks_rust::{benchmarking::ModelEvaluation};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    
    let model_stats = ModelEvaluation::from_json_file(format!("models_stats/{}.json", model_name));

    let mut fg = Figure::new();
    let mut axes = fg.axes2d()
        .set_title("R² over epochs", &[])
        .set_x_label("epochs", &[])
        .set_y_label("R²", &[])
        .set_y_range(Fix(0.8), Fix(1.0));
    
    let colors = &[
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "spring-green"
    ];

    for (i, fold_eval) in model_stats.folds.iter().enumerate() {
        let color = colors[i % colors.len()];

        let x = (0..fold_eval.epochs.len()).collect::<Vec<usize>>();
        let y1 = fold_eval.epochs.iter().map(|epoch_eval| epoch_eval.r2);

        axes = axes
            .points(x.clone(), y1.clone(), &[Color(color), PointSymbol('+'), Caption(&format!("{} fold {} R^2", model_name.replace("_", " "), i))])
    }
    
    fg.save_to_png(format!("visuals/{}_folds_r2.png", model_name), 1024, 728).unwrap();
}