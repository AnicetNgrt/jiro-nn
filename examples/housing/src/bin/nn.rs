use nn::{
    activation::Activation, datatable::DataTable, loss::mse, network::Network, nn_h2,
    benchmarking::Benchmark,
};

const FEATURES: usize = 4;
const OUT: usize = 1;

pub fn new_network() -> Network<FEATURES, OUT> {
    nn_h2::<FEATURES, 30, 15, OUT>(vec![Activation::Tanh])
}

pub fn main() {
    let df = DataTable::from_file("dataset/train.csv");
    let df2 = DataTable::from_file("dataset/test.csv");

    let (test_x, test_y) = df2.random_in_out_batch(&["price"], None);

    let mut network = new_network();

    let mut stats_table = DataTable::new_empty();

    for e in 0..30 {
        let batch_size = 17000 / (300 / (e + 1));
        let (train_x, train_y) = df.random_in_out_batch(&["price"], Some(batch_size));
        network.train(&train_x, &train_y, 0.01, &mse::new());

        let aggregates = Benchmark::from_test_data(&mut network, &test_x, &test_y)
            .compute_all_metrics_aggregates()
            .get_result();

        stats_table = stats_table.apppend(
            aggregates
                .to_datatable(&["price"])
                .with_column_f64("epoch", &[e as f64]),
        );

        println!(
            "[epoch {}] avg dist: {:.6} $",
            e,
            (aggregates.avg.unwrap().dist[0]) * (1_700_000. - 70_000.) + 70_000.
        );
    }

    stats_table.to_file("models_stats/model0.csv");
}
