use housing::{denorm, denorm_single, FEATURES, OUT};
use nn::{
    activation::Activation, benchmarking::Benchmark, datatable::DataTable, loss::mse,
    network::Network, nn,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;

pub fn new_network(hidden_sizes: Vec<usize>, activation: Activation) -> Network {
    let mut sizes = vec![FEATURES];
    for s in hidden_sizes {
        sizes.push(s);
    }
    sizes.push(OUT);
    nn(vec![activation], sizes)
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

#[derive(Serialize, Deserialize, Debug)]
struct GrowingBatchSize {
    start: f64,
    end: f64,
}

#[derive(Serialize, Deserialize)]
struct Config {
    name: String,
    learning_rate: f64,
    epochs: usize,
    decay_rate: Option<f64>,
    dropout: Option<Vec<f64>>,
    layers: Vec<usize>,
    activation: String,
    growing_batch_size: Option<GrowingBatchSize>,
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

    let activation: Activation = match config.activation.as_str() {
        "tanh" => Activation::Tanh,
        "relu" => Activation::ReLU,
        "sigm" => Activation::Sigmoid,
        "hbt" => Activation::HyperbolicTangent,
        _ => panic!("Unknown activation function"),
    };

    // print all hyperparameters
    println!("name: {}", config.name);
    println!("learning_rate: {}", config.learning_rate);
    println!("epochs: {}", config.epochs);
    println!("dropout_rates: {:?}", config.dropout);
    println!("layers_size: {:?}", config.layers);
    println!("activation: {:?}", config.activation);
    println!("decay_rate: {:?}", config.decay_rate);
    println!("growing_batch_size: {:?}", config.growing_batch_size);

    let train_table = DataTable::from_file("dataset/train.csv");
    let test_table = DataTable::from_file("dataset/test.csv");

    let (test_x, test_y) = test_table.random_in_out_batch(&["price"], None);
    let test_y = denorm(&test_y);

    // do the following thing 10 times in parallel


    let mut network = new_network(config.layers, activation);

    let mut stats_table = DataTable::new_empty();

    for e in 0..config.epochs {
        let batch_size = if let Some(ref growing_batch_size) = config.growing_batch_size {
            let progress = smoothstep(e as f64, 0., (config.epochs-1) as f64);
            progress * growing_batch_size.end + (1. - progress) * growing_batch_size.start
        } else {
            1.0
        };
        let batch_size = (batch_size * 17000.) as usize;
        println!("epoch: {} batch_size: {}", e, batch_size);
        let (train_x, train_y) =
            train_table.random_in_out_batch(&["price"], Some(batch_size));
        let learning_rate = if let Some(ref decay_rate) = config.decay_rate {
            config.learning_rate / (1.0 + (decay_rate * e as f64))
        } else {
            config.learning_rate
        };
        if let Some(ref dropout) = config.dropout {
            network.set_dropout_rates(&dropout);
        }
        network.train(&train_x, &train_y, learning_rate, &mse::new());
        network.remove_dropout_rates();

        let preds = network.predict_many(&test_x);
        let preds = &denorm(&preds);

        if e == config.epochs - 1 {
            let test_table = test_table
                .with_column_f64(
                    "predicted price",
                    &preds.iter().map(|x| x[0]).collect::<Vec<f64>>(),
                )
                .map_f64_column("price", denorm_single);
            test_table.to_file(format!("models_preds/{}.csv", config.name));
        }

        let aggregates = Benchmark::from_preds(&preds, &test_y)
            .compute_all_metrics_aggregates()
            .get_result();

        stats_table = stats_table.apppend(
            aggregates
                .to_datatable(&["price"])
                .with_column_f64("epoch", &[e as f64]),
        );

        let prop = aggregates.avg.unwrap().prop_dist[0];
        println!("[epoch {}] avg prop dist: {:.6}", e, prop,);
    }

    stats_table.to_file(format!("models_stats/{}.csv", config.name));
}
