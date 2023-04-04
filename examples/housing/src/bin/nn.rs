use nn::{layer::{full_layer::FullLayer, dense_layer::DenseLayer, hidden_layer::HiddenLayer}, activation::Activation, network::Network, loss::mse, data_utils::DataTable, stats_utils::{PredStats, AggregatedPredStats}};

const PARAMS_IN: usize = 4;
const TRAINING_POINTS: usize = 17290;
const OUT: usize = 1;

pub fn new_network() -> Network<PARAMS_IN, OUT> {
    let layer0 = FullLayer::<PARAMS_IN, 10>::new(
        DenseLayer::new(),
        Activation::Tanh.to_layer()
    );
    let layer1 = FullLayer::<10, 15>::new(
        DenseLayer::new(),
        Activation::Tanh.to_layer()
    );
    let layer2 = FullLayer::<15, OUT>::new(
        DenseLayer::new(),
        Activation::Tanh.to_layer()
    );
    let global = HiddenLayer::new(layer0, layer1, layer2);

    Network::new(Box::new(global))
}

pub fn main() {
    let df = DataTable::from_file("dataset/train.csv");
    let df2 = DataTable::from_file("dataset/test.csv");

    let train_outputs: Vec<_> = df.select_columns(vec!["price"]).transpose().columns_to_vecf64();
    let train_inputs: Vec<_> = df.drop_column("price").transpose().columns_to_vecf64();
    let test_y = df2.select_columns(vec!["price"]).transpose().columns_to_vecf64();
    let test_x = df2.drop_column("price").transpose().columns_to_vecf64();

    println!("{}", train_inputs.len());

    let mut network = new_network();

    let mut errors = vec![];
    for e in 0..100 {
        let error = network.train::<TRAINING_POINTS>(train_inputs.clone(), train_outputs.clone(), 0.1, &mse::new());
        let testset_preds: Vec<_> = network.predict_many(test_x.clone());
        let stats = AggregatedPredStats::new(
            PredStats::many_new(&testset_preds, &test_y), 
            true, true, true, true
        );

        println!("[epoch {}] error: {:.6} | stats: {:#?}", e, error, stats);
        errors.push(error);
    }
}
