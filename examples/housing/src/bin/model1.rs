use std::env;

use housing::{FEATURES, OUT, denorm};
use nn::{
    activation::Activation, benchmarking::Benchmark, datatable::DataTable, loss::mse,
    network::Network, nn_h2,
};

pub fn new_network() -> Network<FEATURES, OUT> {
    nn_h2::<FEATURES, 20, 10, OUT>(vec![Activation::Tanh])
}

pub fn main() {
    let args: Vec<String> = env::args().collect();

    let df = DataTable::from_file("dataset/train.csv");
    let df2 = DataTable::from_file("dataset/test.csv");

    let (test_x, test_y) = df2.random_in_out_batch(&["price"], None);
    let test_y = denorm(&test_y);

    // do this 10 times in parallel and average the aggregates

    let mut network = new_network();

    let mut stats_table = DataTable::new_empty();

    let epochs = args[1].parse::<usize>().unwrap();
    let learning_rate = args[2].parse::<f64>().unwrap();

    for e in 0..epochs {
        let batch_size = (e+1) * (17000/epochs);
        let (train_x, train_y) = df.random_in_out_batch(&["price"], Some(batch_size));
        network.train(&train_x, &train_y, learning_rate, &mse::new());

        let preds = network.predict_many(&test_x);
        let preds = &denorm(&preds);
        let aggregates = Benchmark::from_preds(&preds, &test_y)
            .compute_all_metrics_aggregates()
            .get_result();

        stats_table = stats_table.apppend(
            aggregates
                .to_datatable(&["price"])
                .with_column_f64("epoch", &[e as f64]),
        );

        let prop = aggregates.avg.unwrap().prop_dist[0];
        println!(
            "[epoch {}] avg prop dist: {:.6}",
            e,
            prop,
        );
    }

    let binding = learning_rate.to_string();
    let lr = binding.split('.').collect::<Vec<&str>>()[1];
    
    stats_table.to_file(format!("models_stats/model1_{}_{}.csv", epochs, lr));
}
