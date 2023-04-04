use nn::{
    network::Network, loss::mse
};

fn x() -> Vec<Vec<f64>> {
    vec![
        vec![0., 0.],
        vec![0., 1.],
        vec![1., 0.],
        vec![1., 1.]
    ]
}

fn y() -> Vec<f64> { vec![ 0., 1., 1., 0. ] }

pub fn test_set_accuracy(network: &mut Network<2, 1>) -> f64 {
    x().into_iter().zip(y().into_iter()).map(|(input, output)| {
        let pred = network.predict(input);
        1. - (pred[0] - output).abs()
    }).sum::<f64>() / y().len() as f64
}

pub fn train_and_test<O>(network: &mut Network<2, 1>, epochs: usize, learning_rate: f64, mut lr_optimizer: O) -> (f64, Vec<f64>)
where 
    O: FnMut(usize, f64) -> f64
{
    let mut errors = Vec::new();
    for e in 0..epochs {
        let learning_rate = lr_optimizer(e, learning_rate);
        let error = network.train::<4>(
            x(), 
            y().chunks(1).map(|v| v.to_vec()).collect(), 
            learning_rate,
            &mse::new()
        );
        errors.push(error);
    }

    (*errors.last().unwrap(), errors)
}