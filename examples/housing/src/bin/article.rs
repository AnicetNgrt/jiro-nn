use jiro_nn::dataset::Dataset;
use jiro_nn::dataset::FeatureTags::*;
use jiro_nn::model::ModelBuilder;
use jiro_nn::monitor::TM;
use jiro_nn::preprocessing::map::*;
use jiro_nn::preprocessing::Pipeline;
use jiro_nn::trainers::kfolds::KFolds;

pub fn main() {
    let mut dataset_config = Dataset::from_file("kc_house_data.csv");
    dataset_config
        .remove_features(&["zipcode", "sqft_living15", "sqft_lot15"])
        .tag_feature("id", IsId)
        .tag_feature("date", DateFormat("%Y%m%dT%H%M%S"))
        .tag_feature("date", AddExtractedMonth)
        .tag_feature("date", AddExtractedTimestamp)
        .tag_feature("date", Not(&UsedInModel))
        .tag_feature(
            "yr_renovated",
            Mapped(
                MapSelector::equal_scalar(0.0),
                MapOp::replace_with_feature("yr_built"),
            ),
        )
        .tag_feature("price", Predicted)
        .tag_all(Log10.only(&["sqft_living", "sqft_above", "price"]))
        .tag_all(AddSquared.except(&["price", "date"]).incl_added_features())
        .tag_all(FilterOutliers.except(&["date"]).incl_added_features())
        .tag_all(Normalized.except(&["date"]).incl_added_features());

    TM::start_monitoring();

    let mut pipeline = Pipeline::basic_single_pass();
    let (dataset_config, data) = pipeline
        .load_data("dataset/kc_house_data.csv", Some(&dataset_config))
        .run();

    let hidden_neurons = 22;
    let output_size = 1;

    let model = ModelBuilder::new(dataset_config)
        .neural_network()
			.full_dense(hidden_neurons)
				.relu()
				.adam()
			.end()
			.full_dense(hidden_neurons)
				.relu()
				.adam()
			.end()
			.full_dense(hidden_neurons)
				.relu()
				.adam()
			.end()
			.full_dense(output_size)
				.linear()
				.adam()
			.end()
        .end()
        .batch_size(128)
        .epochs(100)
        .build();

	let mut kfold = KFolds::new(4);

	let (preds_and_ids, model_eval) = kfold
		.compute_best_model()
		.run(&model, &data);
	
	TM::stop_monitoring();

	let best_model_params = kfold.take_best_model();
	best_model_params.to_binary_compressed("best_model_params.gz");

	let preds_and_ids = pipeline.revert(&preds_and_ids);
	let data = pipeline.revert(&data);
	let data_and_preds = data.inner_join(
		&preds_and_ids, 
		"id", "id", 
		Some("pred")
	);

	data_and_preds.to_csv_file("data_and_preds.csv");
	model_eval.to_json_file("model_eval.json");
}
