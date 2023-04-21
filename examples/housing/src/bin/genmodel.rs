use nn::{
    activation::Activation::*,
    dataset::{Dataset, FeatureOptions::*},
    model_spec::{LayerOptions::*, LayerSpec, ModelOptions::*, ModelSpec},
    optimizer::Optimizers,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let dataset = Dataset::from_features_options(
        "kc_house_data",
        &[
            &[
                Name("lat"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("lat^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("long"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("long^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("price"),
                Normalized(true),
                FilterOutliers(true),
                Log10(true),
                Out(true),
            ],
        ],
    );

    let h_size = dataset.in_features_names().len() + 1;
    let nh = 8;

    let mut layers = vec![];
    for _ in 0..nh {
        layers.push(LayerSpec::from_options(&[
            OutSize(h_size),
            Activation(ReLU),
            WeightsOptimizer(Optimizers::adam_default()),
            BiasesOptimizer(Optimizers::adam_default()),
        ]));
    }

    let final_layer = LayerSpec::from_options(&[
        OutSize(1),
        Activation(Linear),
        WeightsOptimizer(Optimizers::adam_default()),
        BiasesOptimizer(Optimizers::adam_default()),
    ]);

    let model = ModelSpec::from_options(&[
        Dataset(dataset),
        HiddenLayers(layers.as_slice()),
        FinalLayer(final_layer),
        BatchSize(Some(128)),
        Folds(8),
        Epochs(500),
    ]);

    println!("{:#?}", model);

    model.to_json_file(format!("models/{}.json", config_name));
}
