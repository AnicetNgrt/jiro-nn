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

pub fn denorm(normalized_ys: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    normalized_ys.iter().map(|normalized_y| {
        normalized_y
            .iter()
            .map(|pred| {
                (10f64).powf((pred * (6.886490725172481 - 4.875061263391699)) + 4.875061263391699)
                // pred * (7700000.0 - 75000.0) + 75000.0
            })
            .collect()
    }).collect()
}