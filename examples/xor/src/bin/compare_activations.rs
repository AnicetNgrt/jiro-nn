use std::thread;

use nn::{nn_h1, activation::Activation};
use xor::score;

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