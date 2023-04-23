use nn::{
    activation::Activation::*,
    dataset::{Dataset, FeatureOptions::*},
    model_spec::{LayerOptions::*, LayerSpec, ModelOptions::*, ModelSpec},
    optimizer::Optimizers,
    pipelines::map::{MapSelector, MapValue, MapOp},
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
                Name("yr_built"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("yr_built^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("yr_renovated"),
                Normalized(true),
                Mapped(
                    MapSelector::Equal(MapValue::ConstantF64("0.0".to_string())),
                    MapOp::Replace(MapValue::Feature("yr_built".to_string())),
                ),
                FilterOutliers(true),
                WithSquared(&[Name("yr_renovated^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("sqft_living"),
                Normalized(true),
                FilterOutliers(true),
                Log10(true),
                WithSquared(&[
                    Name("sqft_living^2"),
                    Normalized(true),
                    FilterOutliers(true),
                ]),
            ],
            &[
                Name("sqft_above"),
                Normalized(true),
                FilterOutliers(true),
                Log10(true),
                WithSquared(&[Name("sqft_above^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("sqft_basement"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[
                    Name("sqft_basement^2"),
                    Normalized(true),
                    FilterOutliers(true),
                ]),
            ],
            &[
                Name("floors"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("floors^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("sqft_lot"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("sqft_lot^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("grade"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("grade^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("bathrooms"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("bathrooms^2"), Normalized(true), FilterOutliers(true)]),
            ],
            &[
                Name("bedrooms"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("bedrooms^2"), Normalized(true)]),
            ],
            &[
                Name("waterfront"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("waterfront^2"), Normalized(true)]),
            ],
            &[
                Name("view"),
                Normalized(true),
                FilterOutliers(true),
                WithSquared(&[Name("view^2"), Normalized(true)]),
            ],
            &[
                Name("date"),
                DateFormat("%Y%m%dT%H%M%S"),
                WithExtractedTimestamp(&[
                    Name("timestamp"),
                    Normalized(true),
                    WithSquared(&[Name("timestamp^2"), Normalized(true)]),
                ]),
                UsedInModel(false),
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
    let dropout = None;

    let mut layers = vec![];
    for i in 0..nh {
        layers.push(LayerSpec::from_options(&[
            OutSize(h_size),
            Activation(ReLU),
            Dropout(if i > 0 { dropout } else { None }),
            WeightsOptimizer(Optimizers::adam_default()),
            BiasesOptimizer(Optimizers::adam_default()),
        ]));
    }

    let final_layer = LayerSpec::from_options(&[
        OutSize(1),
        Activation(Linear),
        Dropout(dropout),
        WeightsOptimizer(Optimizers::adam_default()),
        BiasesOptimizer(Optimizers::adam_default()),
    ]);

    let model = ModelSpec::from_options(&[
        Dataset(dataset),
        HiddenLayers(layers.as_slice()),
        FinalLayer(final_layer),
        BatchSize(Some(128)),
        Folds(8),
        Epochs(300),
    ]);

    println!("{:#?}", model);

    model.to_json_file(format!("models/{}.json", config_name));
}
