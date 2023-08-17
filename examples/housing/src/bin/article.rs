use jiro_nn::dataset::Dataset;
use jiro_nn::dataset::FeatureTags::*;
use jiro_nn::model::ModelBuilder;
use jiro_nn::monitor::TM;
use jiro_nn::preprocessing::map::*;
use jiro_nn::preprocessing::Pipeline;
use jiro_nn::trainers::kfolds::KFolds;

pub fn main() {
    let mut dataset_config = Dataset::from_file("dataset/kc_house_data.csv");
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
        .tag_all(Log10.only(&["sqft_living", "sqft_above"]))
        .tag_all(AddSquared.except(&["price", "date"]).incl_added_features())
        .tag_all(FilterOutliers.except(&["date"]).incl_added_features());
        //.tag_all(Normalized.except(&["date"]).incl_added_features());

    TM::start_monitoring();

    let mut pipeline = Pipeline::basic_single_pass();
    let (dataset_config, data) = pipeline
        .load_data("dataset/kc_house_data.csv", Some(&dataset_config))
        .run();
	
	TM::stop_monitoring();

    data.to_csv_file("dataset/preprocessed.csv");
}
