use std::thread;

use nn::{nn, activation::Activation, network::Network, optimizer::{sgd::SGD, Optimizers}};
use xor::train_and_test;

pub fn score(mut network: &mut Network, epochs: usize, trials: usize) -> f64 {
    (0..trials)
        .map(|_| train_and_test(&mut network, epochs).0)
        .sum::<f64>() / trials as f64
}

fn main() {
    let mut handles = vec![];
    let trials = 10;
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![2, 3, 1], vec![Activation::Tanh], vec![Optimizers::SGD(SGD::with_const_lr(0.1))]);
        let score = score(&mut network, 5000, trials);
        println!("[TANH]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![2, 3, 1], vec![Activation::Sigmoid], vec![Optimizers::SGD(SGD::with_const_lr(0.1))]);
        let score = score(&mut network, 5000, trials);
        println!("[SIGMOID]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![2, 3, 1], vec![Activation::HyperbolicTangent], vec![Optimizers::SGD(SGD::with_const_lr(0.1))]);
        let score = score(&mut network, 5000, trials);
        println!("[HBT]\n avg final error: {}", score);
    }));
    handles.push(thread::spawn(move || {
        let mut network = nn(vec![2, 3, 1], vec![Activation::ReLU], vec![Optimizers::SGD(SGD::with_const_lr(0.1))]);
        let score = score(&mut network, 5000, trials);
        println!("[RELU]\n avg final error: {}", score);
    }));

    for h in handles.into_iter() {
        h.join().unwrap();
    }
}