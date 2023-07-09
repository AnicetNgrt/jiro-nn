use jiro_nn::{
	dataset::{Dataset, FeatureTags::*},
	model::ModelBuilder,
	preprocessing::map::{MapOp, MapSelector, MapValue},
};

fn main() {
	let args: Vec<String> = std::env::args().collect();
	let config_name = &args[1];

	let mut dataset_config = Dataset::from_file("dataset/kc_house_data.csv");
	dataset_config
		.remove_features(&["id", "zipcode", "sqft_living15", "sqft_lot15"])
		.tag_feature("date", DateFormat("%Y%m%dT%H%M%S"))
		.tag_feature("date", AddExtractedMonth)
		.tag_feature("date", AddExtractedTimestamp)
		.tag_feature("date", Not(&UsedInModel))
		.tag_feature(
			"yr_renovated",
			Mapped(
				MapSelector::equal(MapValue::f64(0.0)),
				MapOp::replace_with(MapValue::take_from_feature("yr_built")),
			),
		)
		.tag_feature("price", Predicted)
		.tag_all(Log10.only(&["sqft_living", "sqft_above", "price"]))
		.tag_all(AddSquared.except(&["price", "date"]).incl_added_features())
		//.tag_all(FilterOutliers.except(&["date"]).incl_added_features())
		.tag_all(Normalized.except(&["date"]).incl_added_features());

	let h_size = dataset_config.in_features_names().len() + 1;
	let nh = 8;

	let mut nn = ModelBuilder::new(dataset_config).neural_network();
	for _ in 0..nh {
		nn = nn
			.full_dense(h_size)
			.relu()
			.momentum()
			.end();
	}
	let model = nn
			.full_dense(1)
			.linear()
			.momentum()
			.end()
		.end()
		.batch_size(128)
		.epochs(100)
		.build();

	println!("{:#?}", model);

	model.to_json_file(format!("models/{}.json", config_name));
}
