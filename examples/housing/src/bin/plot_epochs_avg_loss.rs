use gnuplot::{Figure, AxesCommon, Coordinate::Graph, PlotOption::{Color, Caption}};
use nn::{benchmarking::ModelEvaluation};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    let model_eval =
        ModelEvaluation::from_json_file(format!("models_stats/{}.json", model_name));

    let mut fg = Figure::new();
    let x = (0..model_eval.get_n_epochs()).collect::<Vec<usize>>();
    let y1 = model_eval.epochs_avg_train_loss();
    let y2 = model_eval.epochs_std_train_loss();
    let y1_minus_y2 = y1.iter().zip(y2.iter()).map(|(a, b)| a - b).collect::<Vec<f64>>();
    let y1_plus_y2 = y1.iter().zip(y2.iter()).map(|(a, b)| a + b).collect::<Vec<f64>>();

    let y3 = model_eval.epochs_avg_test_loss();
    let y4 = model_eval.epochs_std_test_loss();
    let y3_minus_y4 = y3.iter().zip(y4.iter()).map(|(a, b)| a - b).collect::<Vec<f64>>();
    let y3_plus_y4 = y3.iter().zip(y4.iter()).map(|(a, b)| a + b).collect::<Vec<f64>>();

    fg.axes2d()
        .set_title("Loss over epochs", &[])
        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
        .set_x_label("epochs", &[])
        .set_y_label("loss", &[])
        .lines(x.clone(), y1.clone(), &[Color("blue"), Caption("train loss")])
        .lines(x.clone(), y3.clone(), &[Color("green"), Caption("test loss")])
        .fill_between(x.clone(), y1_minus_y2, y1_plus_y2, &[Color("#AAAAFF"), Caption("train loss +- std")])
        .fill_between(x.clone(), y3_minus_y4, y3_plus_y4, &[Color("#AAFFAA"), Caption("test loss +- std")])
        .set_y_log(Some(10.));
    
    fg.save_to_png(format!("visuals/{}_epochs_avg_loss.png", model_name), 1024, 728).unwrap();

    // Chart::new("Loss over epochs", "loss")
    //     .add_range_x_axis("epochs", 0., model_eval.get_n_epochs() as f64)
    //     .add_discrete_y("avg over folds", model_eval.get_epochs_avg_loss_avg())
    //     .add_discrete_y("std over folds", model_eval.get_epochs_std_loss_avg())
    //     .scatter(format!("visuals/{}_epochs_avg_loss.png", model_name));
}
