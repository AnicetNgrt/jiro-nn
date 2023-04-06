use plotly::color::{Rgb};
use rand::{thread_rng, Rng};

pub const FEATURES: usize = 12;
pub const OUT: usize = 1;
pub const FEATURES_NAMES: &[&str; FEATURES + OUT] = &[
    "price",
    "floors",
    "bedrooms",
    "lat",
    "long",
    "sqft_lot",
    "sqft_living",
    "sqft_above",
    "sqft_basement",
    "grade",
    "condition",
    "yr_built",
    "yr_renovated",
];

pub fn denorm_single(pred: f64) -> f64 {
    (10f64).powf((pred * (6.886490725172481 - 4.875061263391699)) + 4.875061263391699)
}

pub fn denorm(normalized_ys: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    normalized_ys
        .iter()
        .map(|normalized_y| {
            normalized_y
                .iter()
                .map(|pred| {
                    (10f64)
                        .powf((pred * (6.886490725172481 - 4.875061263391699)) + 4.875061263391699)
                })
                .collect()
        })
        .collect()
}

pub fn colors(n: usize) -> Vec<Rgb> {
    let colors = [
        Rgb::new(255, 0, 0),      // red
        Rgb::new(0, 255, 0),      // green
        Rgb::new(0, 0, 255),      // blue
        Rgb::new(255, 128, 0),    // yellow
        Rgb::new(255, 0, 175),    // magenta
        Rgb::new(0, 128, 255),    // cyan
        Rgb::new(255, 0, 128),  // gray
        Rgb::new(128, 0, 0),      // maroon
        Rgb::new(0, 128, 0),      // dark green
        Rgb::new(0, 0, 128),      // navy
    ];

    let mut rng = thread_rng();
    let mut rng_colors = Vec::new();

    for _ in 0..n {
        let mut color = rng.gen_range(0..colors.len());
        while rng_colors.contains(&color) {
            color = rng.gen_range(0..colors.len());
        }
        rng_colors.push(color);
    }

    rng_colors.iter().map(|i| colors[*i]).collect()
}