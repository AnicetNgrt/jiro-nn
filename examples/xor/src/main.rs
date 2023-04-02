use nalgebra::{SMatrix, Vector2, Vector1};
use nn::{
    nn_h1,
    Activation, mean_squared_error::MeanSquaredError
};

fn main() {
    let mut network = nn_h1::<2, 3, 1>(vec![Activation::Tanh]);

    network.fit::<4, MeanSquaredError>(
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
        10000, 
        0.1
    );

    let inputs = &[
        Vector2::new(0., 0.),
        Vector2::new(0., 1.),
        Vector2::new(1., 0.),
        Vector2::new(1., 1.)
    ];

    inputs.iter().for_each(|test| println!("{} => {}", test, network.predict(*test)));
}
