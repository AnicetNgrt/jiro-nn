use neural_networks_rust::{
    dataset::{Dataset, FeatureOptions::*},
    model::ModelBuilder,
    loss::Losses,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut dataset_spec = Dataset::from_csv("dataset/train.csv");
    dataset_spec
        .add_opt(Normalized.except(&["label"]))
        .add_opt_to("label", Out)
        .add_opt_to("label", OneHotEncode);

    let model = ModelBuilder::new(dataset_spec)
        .neural_network()
            .conv_network(1)
                .full_dense(8, 4)
                    .relu()
                    .adam()
                .end()
                .avg_pooling(5)
                .full_direct(2)
                    .relu()
                    .adam()
                .end()
            .end()
            .full_dense(10)
                .softmax()
                .momentum()
            .end()
        .end()
        .epochs(10)
        .batch_size(128)
        .loss(Losses::BCE)
        .build();

    //println!("{:#?}", model);

    model.to_json_file(format!("models/{}.json", config_name));
}
