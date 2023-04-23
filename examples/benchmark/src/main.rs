use std::{ops::Add, time::Instant};

use nn::{
    activation::Activation,
    datatable::DataTable,
    layer::{
        dense_layer::{default_biases_initializer, default_weights_initializer, DenseLayer},
        full_layer::FullLayer,
    },
    linalg::{Matrix, MatrixTrait},
    loss::Losses,
    network::Network,
    optimizer::Optimizers,
};

pub fn f(x: &Vec<f64>) -> Vec<f64> {
    let mut res = vec![0.];

    for i in 0..x.len() {
        res[0] *= (x[i]).sin();
    }

    res
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];
    let epochs = args[2].parse::<usize>().unwrap();

    let input_size = 32;
    let batch_size = 128;
    let batch_count = 100;
    let layer_count = 10;

    let mut sizes = vec![input_size; layer_count + 1];
    sizes.push(1);
    let mut layers = vec![];

    for i in 0..sizes.len() - 1 {
        let in_size = sizes[i];
        let out_size = sizes[i + 1];
        let layer = FullLayer::new(
            DenseLayer::new(
                in_size,
                out_size,
                Optimizers::adam_default(),
                Optimizers::adam_default(),
                default_weights_initializer(),
                default_biases_initializer(),
            ),
            Activation::Sigmoid.to_layer(),
            None,
        );
        layers.push(layer);
    }

    let mut network = Network::new(layers, *sizes.first().unwrap(), *sizes.last().unwrap());

    let train_x = Matrix::random_uniform(input_size, batch_size * batch_count, -1.0, 1.0);
    let train_y = train_x
        .get_data()
        .iter()
        .map(|col| f(col))
        .collect::<Vec<_>>();
    let train_y = Matrix::from_column_leading_matrix(&train_y);
    let train_y = train_y.component_add(&Matrix::random_uniform(
        train_y.dim().0,
        train_y.dim().1,
        -0.1,
        0.1,
    ));

    let test_x = Matrix::random_uniform(input_size, 128, -1.0, 1.0);
    let test_y = test_x
        .get_data()
        .iter()
        .map(|col| f(col))
        .collect::<Vec<_>>();
    let test_y = Matrix::from_column_leading_matrix(&test_y);

    let mut training_times = vec![];
    let mut testing_times = vec![];
    let mut total = vec![];
    let mut eps = vec![];

    for e in 0..epochs {
        eps.push(e as f64);

        println!("epoch: {} starting", e);
        let start = Instant::now();
        let train_loss = network.train(
            0,
            &train_x.get_data(),
            &train_y.get_data(),
            &Losses::MSE.to_loss(),
            batch_size,
        );
        let end = Instant::now();
        training_times.push(end - start);
        println!(
            "epoch: {}, train_loss: {}, time: {:?}",
            e,
            train_loss,
            end - start
        );

        let start = Instant::now();
        let (_preds, loss_avg, _loss_std) = network.predict_evaluate_many(
            &test_x.get_data(),
            &test_y.get_data(),
            &Losses::MSE.to_loss(),
        );
        let end = Instant::now();
        testing_times.push(end - start);
        total.push(
            training_times
                .last()
                .unwrap()
                .add(*testing_times.last().unwrap()),
        );
        println!(
            "epoch: {}, test_loss: {}, time: {:?}",
            e,
            loss_avg,
            end - start
        );
        println!("");
    }

    let time_matrix = &vec![
        eps,
        training_times
            .iter()
            .map(|t| t.as_millis() as f64)
            .collect::<Vec<_>>(),
        testing_times
            .iter()
            .map(|t| t.as_millis() as f64)
            .collect::<Vec<_>>(),
        total
            .iter()
            .map(|t| t.as_millis() as f64)
            .collect::<Vec<_>>(),
    ];
    let time_matrix = Matrix::from_column_leading_matrix(&time_matrix.clone());
    let time_matrix = time_matrix.get_data_row_leading();

    let datatable = DataTable::from_vectors(
        &["epoch", "training_time", "testing_time", "total_time"],
        &time_matrix,
    );

    datatable.to_file(format!(
        "results/{}_{}ep_{}params_{}x{}in.csv",
        config_name, epochs, input_size, batch_size, batch_count
    ));
}
