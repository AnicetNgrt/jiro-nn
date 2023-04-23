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

pub fn test_set_accuracy(network: &mut Network) -> f64 {
    x().into_iter().zip(y().into_iter()).map(|(input, output)| {
        let pred = network.predict(input);
        1. - (pred[0] - output).abs()
    }).sum::<f64>() / y().len() as f64
}

pub fn train_and_test(network: &mut Network, epochs: usize) -> (f64, Vec<f64>)
{
    let mut errors = Vec::new();
    for e in 0..epochs {
        let error = network.train(
            e,
            &x(), 
            &y().chunks(1).map(|v| v.to_vec()).collect(), 
            &mse::new(),
            1
        );
        errors.push(error);
    }

    (*errors.last().unwrap(), errors)
}