use jiro_nn::{dataset::{Dataset, FeatureTags}, preprocessing::Pipeline, model::ModelBuilder, loss::Losses, trainers::kfolds::KFolds};

pub fn main() {
    let dataset_path = "mnist.parquet";

    // preprocessing without messing with polars
    let mut dataset_config = Dataset::from_file(dataset_path);
    let dataset_config = dataset_config
        .tag_feature("id", FeatureTags::IsId)
        .tag_feature("label", FeatureTags::Predicted)
        .tag_feature("label", FeatureTags::OneHotEncode)
        .tag_all(FeatureTags::Normalized.except(&["label", "id"]));

    let mut pipeline = Pipeline::basic_single_pass();
    let (dataset_config, data) = pipeline
        .load_data(dataset_path, Some(dataset_config))
        .run();

    // model building without looking everywhere for the right structs
    let model = ModelBuilder::new(dataset_config)
        .neural_network()
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

    // training without installing a dedicated k-folds crate 
    // nor messing with hairy tensors
    let mut kfolds = KFolds::new(10);
    let (predictions_by_id, model_eval) = kfolds.run(&model, &data);

    // saving the model
    model_eval.to_json_file("mnist_eval.json");
    kfolds.take_best_model().to_json("mnist_weights.json");

    // saving the predictions alongside the original data
    let predictions_by_id = pipeline.revert(&predictions_by_id);
    pipeline.revert(&data)
        .inner_join(&predictions_by_id, "id", "id", Some("pred"))
        .to_parquet_file("mnist_values_and_preds.parquet");
}