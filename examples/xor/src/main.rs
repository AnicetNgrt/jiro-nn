use std::thread;

use nalgebra::{SMatrix, Vector2, Vector1};
use nn::{
    nn_h1,
    network::Network,
    activation::Activation, loss::mse::MeanSquaredError
};

fn train_and_test(network: &mut Network<2, 1>, epochs: usize, learning_rate: f64) -> f64 {
    let errors = network.fit::<4, MeanSquaredError>(
        SMatrix::from_columns(&[
            Vector2::new(0., 0.),
            Vector2::new(0., 1.),
            Vector2::new(1., 0.),
            Vector2::new(1., 1.)
        ]), 
        SMatrix::from_columns(&[
            Vector1::new(0.),
            Vector1::new(1.),
            Vector1::new(1.),
            Vector1::new(0.)
        ]), 
        epochs, 
        learning_rate
    );

    *errors.last().unwrap()
}

fn score(mut network: &mut Network<2, 1>, epochs: usize, learning_rate: f64, trials: usize) -> f64 {
    (0..trials)
        .map(|_| train_and_test(&mut network, epochs, learning_rate))
        .sum::<f64>() / trials as f64
}

fn main() {
    let mut handles = vec![];
    let trials = 10;
    handles.push(thread::spawn(move || {
        let mut network = nn_h1::<2, 3, 1>(vec![Activation::Tanh]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[TANH]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn_h1::<2, 3, 1>(vec![Activation::Sigmoid]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[SIGMOID]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn_h1::<2, 3, 1>(vec![Activation::HyperbolicTangent]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[HBT]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn_h1::<2, 3, 1>(vec![Activation::ReLU]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[RELU]\n avg final error: {}", score);
    }));

    for h in handles.into_iter() {
        h.join().unwrap();
    }
}
