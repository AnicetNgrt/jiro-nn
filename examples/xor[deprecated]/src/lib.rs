use nn::{
    network::Network, loss::mse
};

fn x() -> Vec<Vec<Scalar>> {
    vec![
        vec![0., 0.],
        vec![0., 1.],
        vec![1., 0.],
        vec![1., 1.]
    ]
}

fn y() -> Vec<Scalar> { vec![ 0., 1., 1., 0. ] }

pub fn test_set_accuracy(network: &mut Network) -> Scalar {
    x().into_iter().zip(y().into_iter()).map(|(input, output)| {
        let pred = network.predict(input);
        1. - (pred[0] - output).abs()
    }).sum::<Scalar>() / y().len() as Scalar
}

pub fn train_and_test(network: &mut Network, epochs: usize) -> (Scalar, Vec<Scalar>)
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