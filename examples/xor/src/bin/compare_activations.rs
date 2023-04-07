use std::thread;

use nn::{nn, activation::Activation, network::Network};
use xor::train_and_test;

pub fn score(mut network: &mut Network, epochs: usize, learning_rate: f64, trials: usize) -> f64 {
    (0..trials)
        .map(|_| train_and_test(&mut network, epochs, learning_rate, |_, lr| lr).0)
        .sum::<f64>() / trials as f64
}

fn main() {
    let mut handles = vec![];
    let trials = 10;
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![Activation::Tanh], vec![2, 3, 1]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[TANH]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![Activation::Sigmoid], vec![2, 3, 1]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[SIGMOID]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![Activation::HyperbolicTangent], vec![2, 3, 1]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[HBT]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![Activation::ReLU], vec![2, 3, 1]);
        let score = score(&mut network, 5000, 0.1, trials);
        println!("[RELU]\n avg final error: {}", score);
    }));

    for h in handles.into_iter() {
        h.join().unwrap();
    }
}