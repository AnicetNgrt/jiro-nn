use std::env;

use housing::{denorm, FEATURES, OUT, denorm_single};
use nn::{activation::Activation, benchmarking::Benchmark, datatable::DataTable, loss::mse, nn_h2};

pub fn main() {
    let args: Vec<String> = env::args().collect();

    let train_table = DataTable::from_file("dataset/train.csv");
    let test_table = DataTable::from_file("dataset/test.csv");

    let (test_x, test_y) = test_table.random_in_out_batch(&["price"], None);
    let test_y = denorm(&test_y);

    let (mut network, configs) = nn_h2::<FEATURES, 20, 10, OUT>(vec![Activation::Tanh]);

    let mut stats_table = DataTable::new_empty();

    let epochs = args[1].parse::<usize>().unwrap();
    let learning_rate = args[2].parse::<f64>().unwrap();
    let dropout_rate_l0 = args[3].parse::<f64>().unwrap();
    let dropout_rate_l1 = args[4].parse::<f64>().unwrap();
    let dropout_rate_l2 = args[5].parse::<f64>().unwrap();

    let binding = learning_rate.to_string();
    let lr = binding.split('.').collect::<Vec<&str>>()[1];
    let model_name = format!(
        "model4_{}_{}_{:03}{:03}{:03}",
        epochs,
        lr,
        (dropout_rate_l0 * 100.) as usize,
        (dropout_rate_l1 * 100.) as usize,
        (dropout_rate_l2 * 100.) as usize
    );

    for e in 0..epochs {
        let batch_size = 17000;
        let (train_x, train_y) = train_table.random_in_out_batch(&["price"], Some(batch_size));

        configs.0.set_dropout_rate(dropout_rate_l0);
        configs.1.set_dropout_rate(dropout_rate_l1);
        configs.2.set_dropout_rate(dropout_rate_l2);
        network.train(
            &train_x,
            &train_y,
            learning_rate / (e as f64 + 1.),
            &mse::new(),
        );

        configs.0.remove_dropout_rate();
        configs.1.remove_dropout_rate();
        configs.2.remove_dropout_rate();
        let preds = network.predict_many(&test_x);
        let preds = &denorm(&preds);

        if e == epochs-1 {
            let test_table = test_table
                .with_column_f64("predicted price", &preds.iter().map(|x| x[0]).collect::<Vec<f64>>())
                .map_f64_column("price", denorm_single);
            test_table.to_file(format!("models_preds/{}.csv", model_name));
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

    stats_table.to_file(format!("models_stats/{}.csv", model_name));
}
