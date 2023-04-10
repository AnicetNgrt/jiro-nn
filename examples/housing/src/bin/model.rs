use housing::{denorm, denorm_single, FEATURES, OUT};
use nn::optimizer::{Optimizers};
use nn::{
    activation::Activation, benchmarking::Benchmark, datatable::DataTable, loss::mse,
    network::Network, nn,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;

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

#[derive(Serialize, Deserialize)]
struct Config {
    name: String,
    epochs: usize,
    dropout: Option<Vec<f64>>,
    layers: Vec<usize>,
    activation: Activation,
    optimizer: Optimizers,
    batch_size: Option<usize>,
}

pub fn main() {
    // get file name from command line
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    // Read the file into a string
    let mut file = File::open(format!("models/{}.json", config_name)).expect("Failed to open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read file");

    // Parse the JSON string into a Config struct
    let config: Config = serde_json::from_str(&contents).expect("Failed to parse JSON");

    // print all hyperparameters
    println!("name: {}", config.name);
    println!("epochs: {}", config.epochs);
    println!("dropout_rates: {:#?}", config.dropout);
    println!("layers_size: {:#?}", config.layers);
    println!("activation: {:#?}", config.activation);
    println!("batch_size: {:#?}", config.batch_size);
    println!("optimizer: {:#?}", config.optimizer);

    let train_table = DataTable::from_file("dataset/train.csv");
    let test_table = DataTable::from_file("dataset/test.csv");

    let (test_x, test_y) = test_table.random_in_out_batch(&["price"], None);
    let denorm_test_y = denorm(&test_y);

    let mut network = new_network(config.layers, config.activation, config.optimizer);

    let mut stats_table = DataTable::new_empty();

    for e in 0..config.epochs {
        let (train_x, train_y) =
        train_table.random_in_out_batch(&["price"], None);
        if let Some(ref dropout) = config.dropout {
            network.set_dropout_rates(&dropout);
        }
        let error = network.train(e, &train_x, &train_y, &mse::new(), config.batch_size.unwrap_or(1));
        network.remove_dropout_rates();

        let preds = network.predict_many(&test_x);
        let denorm_preds = &denorm(&preds);

        if e == config.epochs - 1 {
            let test_table = test_table
                .with_column_f64(
                    "predicted price",
                    &denorm_preds.iter().map(|x| x[0]).collect::<Vec<f64>>(),
                )
                .map_f64_column("price", denorm_single);
            test_table.to_file(format!("models_preds/{}.csv", config.name));
        }

        let aggregates = Benchmark::from_preds(&preds, &test_y)
            .compute_all_metrics_aggregates()
            .get_result();

        let denorm_aggregates = Benchmark::from_preds(&denorm_preds, &denorm_test_y)
            .compute_all_metrics_aggregates()
            .get_result();

        stats_table = stats_table.apppend(
            denorm_aggregates
                .to_datatable(&["price"])
                .with_column_f64("epoch", &[e as f64]),
        );

        let prop = denorm_aggregates.avg.unwrap().prop_dist[0];
        let mse = aggregates.avg.unwrap().mse;
        println!("[epoch {}] training avg mse: {:.6} test avg mse: {:.6} denorm test avg prop dist: {:.6}", e, error, mse, prop);
    }

    stats_table.to_file(format!("models_stats/{}.csv", config.name));
}
