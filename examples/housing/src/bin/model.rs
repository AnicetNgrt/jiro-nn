use housing::{FEATURES, OUT};
use nn::model_spec::ModelSpec;
use nn::optimizer::{Optimizers};
use nn::pipelines::Pipeline;
use nn::pipelines::log_scale::LogScale10;
use nn::pipelines::normalize::Normalize;
use nn::pipelines::to_timestamp::ToTimestamps;
use nn::{
    activation::Activation,
    network::Network, nn,
};

pub fn new_network(hidden_sizes: Vec<usize>, activation: Activation, optimizer: Optimizers) -> Network {
    let mut sizes = vec![FEATURES];
    for s in hidden_sizes {
        sizes.push(s);
    }
    sizes.push(OUT);
    nn(sizes, vec![activation], vec![optimizer])
}

pub fn smoothstep(x: f64, min: f64, max: f64) -> f64 {
    if x <= min {
        0.
    } else if x >= max {
        1.
    } else {
        (x - min) / (max - min)
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let model = ModelSpec::from_json_file(format!("models/{}.json", config_name).as_str());
    println!("model: {:#?}", model);

    let mut pipeline = Pipeline::new();
    let (dataset_spec, data) = pipeline
        .add(Box::new(ToTimestamps))
        .add(Box::new(LogScale10))
        .add(Box::new(Normalize))
        .run("dataset", &model.dataset);

    println!("dataset: {:#?}", dataset_spec);
    println!("data: {:#?}", data);

    // let (train_table, test_table) = data
    //     .sample(Some(TRAINING_POINTS + TESTING_POINTS), true)
    //     .select_columns(FEATURES_NAMES)
    //     .split(TRAINING_POINTS, TESTING_POINTS);

    // let serialized_config = serde_json::to_string(&config).unwrap();
    // train_table.with_column_str("model", &[&serialized_config]).to_file("dataset/train.csv");
    // test_table.with_column_str("model", &[&serialized_config]).to_file("dataset/test.csv");


    // let train_table = DataTable::from_file("dataset/train.csv");
    // let test_table = DataTable::from_file("dataset/test.csv");

    // let (test_x, test_y) = test_table.random_in_out_batch(&["price"], None);
    // let denorm_test_y = denorm(&test_y);

    // let mut network = new_network(config.layers, config.activation, config.optimizer);

    // let mut stats_table = DataTable::new_empty();

    // for e in 0..config.epochs {
    //     let (train_x, train_y) =
    //     train_table.random_in_out_batch(&["price"], None);
    //     if let Some(ref dropout) = config.dropout {
    //         network.set_dropout_rates(&dropout);
    //     }
    //     let error = network.train(e, &train_x, &train_y, &mse::new(), config.batch_size.unwrap_or(1));
    //     network.remove_dropout_rates();

    //     let preds = network.predict_many(&test_x);
    //     let denorm_preds = &denorm(&preds);

    //     if e == config.epochs - 1 {
    //         let test_table = test_table
    //             .with_column_f64(
    //                 "predicted price",
    //                 &denorm_preds.iter().map(|x| x[0]).collect::<Vec<f64>>(),
    //             )
    //             .map_f64_column("price", denorm_single);
    //         test_table.to_file(format!("models_preds/{}.csv", config.name));
    //     }

    //     let aggregates = Benchmark::from_preds(&preds, &test_y)
    //         .compute_all_metrics_aggregates()
    //         .get_result();

    //     let denorm_aggregates = Benchmark::from_preds(&denorm_preds, &denorm_test_y)
    //         .compute_all_metrics_aggregates()
    //         .get_result();

    //     stats_table = stats_table.apppend(
    //         denorm_aggregates
    //             .to_datatable(&["price"])
    //             .with_column_f64("epoch", &[e as f64]),
    //     );

    //     let prop = denorm_aggregates.avg.unwrap().prop_dist[0];
    //     let mse = aggregates.avg.unwrap().mse;
    //     println!("[epoch {}] training avg mse: {:.6} test avg mse: {:.6} denorm test avg prop dist: {:.6}", e, error, mse, prop);
    // }

    // stats_table.to_file(format!("models_stats/{}.csv", config.name));
}
