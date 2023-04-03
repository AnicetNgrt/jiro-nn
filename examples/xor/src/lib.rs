use nn::{
    network::Network, loss::mse
};

pub fn train_and_test(network: &mut Network<2, 1>, epochs: usize, learning_rate: f64) -> f64 {
    let errors = network.fit_iter::<4, _>(
        vec![
            0., 0.,
            0., 1.,
            1., 0.,
            1., 1.
        ], 
        vec![ 0., 1., 1., 0. ], 
        epochs, 
        learning_rate,
        mse::new()
    );

    *errors.last().unwrap()
}

pub fn score(mut network: &mut Network<2, 1>, epochs: usize, learning_rate: f64, trials: usize) -> f64 {
    (0..trials)
        .map(|_| train_and_test(&mut network, epochs, learning_rate))
        .sum::<f64>() / trials as f64
}