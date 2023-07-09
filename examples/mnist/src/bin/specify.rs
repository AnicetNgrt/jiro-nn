use jiro_nn::{
    dataset::{Dataset, FeatureTags::*},
    model::ModelBuilder,
    loss::Losses,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_name = &args[1];

    let mut dataset_config = Dataset::from_file("dataset/train_cleaned.parquet");
    dataset_config
        .tag_all(Normalized.except(&["label"]))
        .tag_feature("id", IsId)
        .tag_feature("label", Predicted)
        .tag_feature("label", OneHotEncode);

    let model = ModelBuilder::new(dataset_config)
        .neural_network()
            // 28x28 pixels in
            .conv_network(1)
                .full_dense(32, 5)
                    .relu()
                    .adam()
                    .dropout(0.4)
                .end()
                .avg_pooling(2)
                .full_dense(64, 5)
                    .relu()
                    .adam()
                    .dropout(0.5)
                .end()
                .avg_pooling(2)
            .end()
            .full_dense(128)
                .relu()
                .adam()
            .end()
            .full_dense(10)
                .softmax()
                .adam()
            .end()
        .end()
        .epochs(20)
        .batch_size(128)
        .loss(Losses::BCE)
        .build();

    //println!("{:#?}", model);

    model.to_json_file(format!("models/{}.json", config_name));
}
