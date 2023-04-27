use neural_networks_rust::{
    activation::Activation::*,
    dataset::{Dataset, FeatureOptions::*},
    model::{LayerOptions::*, LayerSpec, ModelOptions::*, Model},
    optimizer::{adam},
    pipelines::map::{MapOp, MapSelector, MapValue}, trainers::Trainers, initializers::Initializers,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut dataset_spec = Dataset::from_csv("dataset/kc_house_data.csv");
    dataset_spec
        .remove_features(&["id", "zipcode", "sqft_living15", "sqft_lot15"])
        .add_opt_to("date", DateFormat("%Y%m%dT%H%M%S"))
        .add_opt_to("date", AddExtractedMonth)
        .add_opt_to("date", AddExtractedTimestamp)
        .add_opt_to("date", Not(&UsedInModel))
        .add_opt_to(
            "yr_renovated",
            Mapped(
                MapSelector::Equal(0.0.into()),
                MapOp::ReplaceWith(MapValue::Feature("yr_built".to_string())),
            ),
        )
        .add_opt_to("price", Out)
        .add_opt(Log10.only(&["sqft_living", "sqft_above", "price"]))
        .add_opt(AddSquared.except(&["price", "date"]).incl_added_features())
        .add_opt(FilterOutliers.except(&["date"]).incl_added_features())
        .add_opt(Normalized.except(&["date"]).incl_added_features());

    let h_size = dataset_spec.in_features_names().len() + 1;
    let nh = 8;
    let dropout = None;

    let mut layers = vec![];
    for i in 0..nh {
        layers.push(LayerSpec::from_options(&[
            OutSize(h_size),
            Activation(ReLU),
            Dropout(if i > 0 { dropout } else { None }),
            Optimizer(adam()),
        ]));
    }

    let final_layer = LayerSpec::from_options(&[
        OutSize(1),
        Activation(Linear),
        Dropout(dropout),
        Optimizer(adam()),
    ]);

    let model = Model::from_options(&[
        Dataset(dataset_spec),
        HiddenLayers(layers.as_slice()),
        FinalLayer(final_layer),
        BatchSize(128),
        Trainer(Trainers::KFolds(8)),
        Epochs(300),
    ]);

    //println!("{:#?}", model);

    model.to_json_file(format!("models/{}.json", config_name));
}
