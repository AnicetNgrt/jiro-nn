use neural_networks_rust::{
	dataset::{Dataset, FeatureOptions::*},
	model::ModelBuilder,
	pipelines::map::{MapOp, MapSelector, MapValue},
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
		//.add_opt(FilterOutliers.except(&["date"]).incl_added_features())
		.add_opt(Normalized.except(&["date"]).incl_added_features());

	let h_size = dataset_spec.in_features_names().len() + 1;
	let nh = 8;

	let mut nn = ModelBuilder::new(dataset_spec).neural_network();
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
		.epochs(500)
		.build();

	println!("{:#?}", model);

	model.to_json_file(format!("models/{}.json", config_name));
}
