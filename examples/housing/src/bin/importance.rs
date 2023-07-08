use gnuplot::*;
use jiro_nn::{
    linalg::Scalar,
    model::Model,
    network::params::NetworkParams,
    preprocessing::{sample::Sample, Pipeline},
    vec_utils::{avg_vector, r2_score_vec2, shuffle_column},
};

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];
    let weights_file = &args[2];

    let mut model = Model::from_json_file(format!("models/{}.json", config_name));

    let mut pipeline = Pipeline::basic_single_pass();
    let (updated_dataset_spec, data) = pipeline
        .prepend(Sample::new(21000, true))
        .load_data_and_spec("./dataset/kc_house_data.csv", &model.dataset)
        .run();

    println!("Data: {:#?}", data);

    let model = model.with_new_dataset(updated_dataset_spec);
    let predicted_features = model.dataset.predicted_features_names();

    let (x_table, y_table) = data.random_order_in_out(&predicted_features);

    let x = x_table.to_vectors();
    let y = y_table.to_vectors();

    let weights = NetworkParams::from_json(format!("models_weights/{}.json", weights_file));
    let mut network = model.to_network();
    network.load_params(&weights);

    let preds = network.predict_many(&x, 1);

    let ref_score = r2_score_vec2(&y, &preds);

    println!("r2: {:#?}", ref_score);

    let mut x_cp = x.clone();

    // Based on https://arxiv.org/pdf/1801.01489.pdf
    // and https://christophm.github.io/interpretable-ml-book/feature-importance.html

    let shuffles_count = 10;
    let ncols = x_cp[0].len();
    let mut means_list = Vec::new();

    for c in 0..ncols {
        println!("Shuffling column {}/{}", c + 1, ncols);

        let mut metric_list = Vec::new();
        for _ in 0..shuffles_count {
            shuffle_column(&mut x_cp, c);
            let preds = network.predict_many(&x_cp, 1);
            let score = r2_score_vec2(&y, &preds);
            println!("score: {:#?}", score);
            metric_list.push(ref_score - score);
            x_cp = x.clone();
        }
        means_list.push(avg_vector(&metric_list));
    }

    // Converting it all to percentages and sorting
    let mut importance_rel = Vec::new();
    let columns_names = x_table.get_columns_names();

    let means_sum = means_list.iter().sum::<Scalar>();
    for (mean, name) in means_list.iter().zip(columns_names.iter()) {
        importance_rel.push(((100.0 * mean) / means_sum, name));
    }
    importance_rel.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut fg = Figure::new();
    let axes = fg
        .axes2d()
        .set_y_range(Fix(0.0), Auto)
        .set_x_range(Fix(-0.5), Fix(importance_rel.len() as f64 + 1.0))
        .set_title("Relative importance (%) of all the features", &[])
        .set_margins(&[MarginBottom(0.24)])
        .set_x_ticks(None, &[], &[])
        .set_y_label("importance (%)", &[]);

    axes.boxes(
        (0i32..(columns_names.len() as i32)).collect::<Vec<i32>>(),
        importance_rel.clone().into_iter().map(|(v, _)| v).collect::<Vec<_>>(),
        &[],
    );

    for i in 0..importance_rel.len() {
        println!("{}: {}", importance_rel[i].1, importance_rel[i].0);
        axes.label(
            &importance_rel[i].1.replace("_", " "),
            Coordinate::Axis(i as f64),
            Coordinate::Axis(-0.2),
            &[Rotate(-40.0)],
        );
    }

    fg.save_to_png(
        format!("visuals/{}_rel_importance.png", config_name),
        1524,
        728,
    )
    .unwrap();
}
