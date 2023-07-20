use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_aux::field_attributes::bool_true;

use crate::{
    datatable::DataTable,
    preprocessing::map::{MapOp, MapSelector},
};

/// A structure that configurationifies _features_ (aka "columns") that will be fed to the network
/// and the preprocessing pipeline. **This is not the actual data**, it is only metadata for the framework.
///
/// Features are described via the `Feature` struct. See its documentation for more information.
///
/// Example:
///
/// ```rust
/// // This will create a Dataset with default metadata for all of the spreadsheet's columns
/// let mut dataset_config = Dataset::from_file("data.csv");
///
/// dataset_config
///     // Adds a "tag" to all features via `tag_all`
///     // The tag used here `Normalized` tells the framework that all the columns
///     // that have it will need to be normalized in preprocessing pipeline later.
///     // The tags can be modified with methods such as `except`, which tells to
///     // skip applying the tag for the "label" column.
///     .tag_all(Normalized.except(&["label"]))
///     // Tags don't always mean preprocessing. Here we simply indicate which column
///     // can be used as an id. Which is important for some parts of the framework.
///     // Here `tag_feature` applies the tag to only one feature.
///     .tag_feature("id", IsId)
///     // Other tags examples below:
///     .tag_feature("label", Predicted)
///     .tag_feature("label", OneHotEncode);
/// ```
///
/// More in-depth example:
///
/// ```rust
/// let mut dataset_config = Dataset::from_file("dataset/kc_house_data.csv");
/// dataset_config
///     .remove_features(&["id", "zipcode", "sqft_living15", "sqft_lot15"])
///     .tag_feature("date", DateFormat("%Y%m%dT%H%M%S"))
///     .tag_feature("date", AddExtractedMonth)
///     .tag_feature("date", AddExtractedTimestamp)
///     .tag_feature("date", Not(&UsedInModel))
///     .tag_feature(
///         "yr_renovated",
///         Mapped(
///             MapSelector::equal(MapValue::f64(0.0)),
///             MapOp::replace_with(MapValue::take_from_feature("yr_built")),
///         ),
///     )
///     .tag_feature("price", Predicted)
///     .tag_all(Log10.only(&["sqft_living", "sqft_above", "price"]))
///     .tag_all(AddSquared.except(&["price", "date"]).incl_added_features())
///     .tag_all(FilterOutliers.except(&["date"]).incl_added_features())
///     .tag_all(Normalized.except(&["date"]).incl_added_features());
/// ```
#[derive(Serialize, Debug, Deserialize, Clone, Hash, Default)]
pub struct Dataset {
    pub features: Vec<Feature>,
}

impl Dataset {
    /// Create a new dataset with a new feature added.
    pub fn with_added_feature(&self, feature: Feature) -> Self {
        let mut features = self.features.clone();
        features.push(feature);
        Self { features }
    }

    pub fn without_feature(&self, feature_name: String) -> Self {
        let mut features = self.features.clone();
        features.retain(|f| f.name != feature_name);
        Self { features }
    }

    /// Create a new dataset with a feature replaced.
    pub fn with_replaced_feature(&self, old_feature_name: &str, feature: Feature) -> Self {
        let mut features = self.features.clone();
        let index = features
            .iter()
            .position(|f| f.name == old_feature_name)
            .unwrap();
        features[index] = feature;
        Self { features }
    }

    pub fn feature_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            names.push(feature.name.as_str());
        }
        names
    }

    /// Return the names of the features that are not configurationified as outputs.
    pub fn in_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if !feature.predicted && !feature.is_id && !feature.date_format.is_some() {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    /// Return the names of the features that are configurationified as outputs.
    pub fn predicted_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if feature.predicted {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    pub fn new(features: &[Feature]) -> Self {
        Self {
            features: features.to_vec(),
        }
    }

    /// Create a new dataset from a data file (csv, parquet...) by adding all its columns as default features.
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        let feature_names = DataTable::columns_names_from_file(path);
        let mut features = Vec::new();
        for feature_name in feature_names {
            let feature = Feature::from_tags(&[FeatureTags::Name(feature_name.as_str())]);
            features.push(feature);
        }
        Self::new(&features)
    }

    pub fn remove_features(&mut self, feature_names: &[&str]) -> &mut Self {
        let mut new_features = Vec::new();
        for feature in &self.features {
            if !feature_names.contains(&feature.name.as_str()) {
                new_features.push(feature.clone());
            }
        }
        self.features = new_features;
        self
    }

    /// The `tag_feature` is a way to tag a configurationific dataset's feature with a `FeatureTags`, in order to configurationify its properties and preprocessing requirements.
    ///
    /// See the `FeatureTags` documentation for more information on the available tags.
    pub fn tag_feature(&mut self, feature_name: &str, tag: FeatureTags) -> &mut Self {
        for feature in &mut self.features {
            if feature.name == feature_name {
                tag.apply(feature);
            }
        }
        self
    }

    /// The `tag_all` method adds a `FeatureTags` to all features in the dataset.
    pub fn tag_all(&mut self, tag: FeatureTags) -> &mut Self {
        for feature in &mut self.features {
            tag.apply(feature);
        }
        self
    }

    /// The `from_features_tags` method is a constructor function for creating a `Dataset` object from a collection of `FeatureTags`.
    ///
    /// This method takes in a name for the dataset, as well as a collection of collections of `FeatureTags` representing the individual features to be included in the dataset. The `FeatureTags` for each feature configurationify how that feature should be preprocessed before being included in the dataset.
    ///
    /// See the `FeatureTags` documentation for more information on the available tags.
    ///
    /// Example:
    ///
    /// ```
    /// let features1 = &[
    ///     FeatureTags::Name("age"),
    ///     FeatureTags::Normalized(true),
    ///     FeatureTags::Squared(true),
    /// ];
    /// let features2 = &[
    ///     FeatureTags::Name("income"),
    ///     FeatureTags::Log10(true),
    ///     FeatureTags::FilterOutliers(true),
    /// ];
    ///
    /// let dataset = from_features_tags("my_dataset", &[&features1, &features2]);
    /// ```

    pub fn from_features_tags(features: &[&[FeatureTags]]) -> Self {
        let mut dataset = Self::default();
        for feature_tags in features {
            let feature = Feature::from_tags(feature_tags);
            dataset.features.push(feature);
        }
        dataset
    }

    pub fn get_id_column(&self) -> Option<&str> {
        for feature in &self.features {
            if feature.is_id {
                return Some(feature.name.as_str());
            }
        }
        None
    }
}

/// A structure that holds metadata of a _feature_ (aka. a "column") of a data table.
#[derive(Default, Serialize, Debug, Deserialize, Clone, Hash, Eq, PartialEq)]
pub struct Feature {
    pub name: String,
    #[serde(default)]
    pub predicted: bool,
    pub date_format: Option<String>,
    #[serde(default)]
    pub to_timestamp: bool,
    #[serde(default)]
    pub extract_month: bool,
    #[serde(default)]
    pub log10: bool,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub filter_outliers: bool,
    #[serde(default)]
    pub mapped: Option<(MapSelector, MapOp)>,
    #[serde(default)]
    pub squared: bool,
    pub with_extracted_timestamp: Option<Box<Feature>>,
    pub with_extracted_month: Option<Box<Feature>>,
    pub with_log10: Option<Box<Feature>>,
    pub with_normalized: Option<Box<Feature>>,
    pub with_squared: Option<Box<Feature>>,
    #[serde(default = "bool_true")]
    pub used_in_model: bool,
    #[serde(default)]
    pub one_hot_encoded: bool,
    #[serde(default)]
    pub is_id: bool,
}

impl Feature {
    /// The `from_tags` method is a constructor function for creating a `Feature` object from a list of `FeatureTags`.
    ///
    /// See the documentation for `FeatureTags` for more information on available tags.

    pub fn from_tags(feature_tags: &[FeatureTags]) -> Self {
        let mut feature = Feature::default();
        feature.used_in_model = true;
        for feature_tag in feature_tags {
            feature_tag.apply(&mut feature);
        }
        feature
    }

    pub fn get_extracted_features_mut(&mut self) -> Vec<&mut Feature> {
        let mut extracted_features = Vec::new();
        if let Some(ref mut feature) = self.with_extracted_month {
            extracted_features.push(feature.as_mut());
        }
        if let Some(ref mut feature) = self.with_extracted_timestamp {
            extracted_features.push(feature.as_mut());
        }
        if let Some(ref mut feature) = self.with_log10 {
            extracted_features.push(feature.as_mut());
        }
        if let Some(ref mut feature) = self.with_normalized {
            extracted_features.push(feature.as_mut());
        }
        if let Some(ref mut feature) = self.with_squared {
            extracted_features.push(feature.as_mut());
        }
        extracted_features
    }
}

/// Tags to be used to describe a feature and its related preprocessing.
///
/// **Feature description tags**:
///
/// - `Name`: The name of the feature.
/// - `UsedInModel`: Disables the pruning of the feature at the end of the pipeline. Features are all not pruned by default.
/// - `Predicted`: Sets the feature as a _predicted feature_. Features are all not _predicted features_ by default.
/// - `IsId`: Identifies the feature as an id. Features are all not ids by default.
/// - `DateFormat`: The date format to use for date/time features.
///
/// **Feature replacement/mapping tags**:
///
/// - `ToTimestamp`: Enables conversion of the date/time feature to a Unix timestamp. Requires the feature to have a `DateFormat` configurationified.
/// - `ExtractMonth`: Enables conversion of the date/time to its month. Requires the feature to have a `DateFormat` configurationified.
/// - `OneHotEncode`: Enables one-hot encoding of the feature.
/// - `Log10`: Enables applying base-10 logarithm to the feature.
/// - `Normalized`: Enables normalizing the feature.
/// - `FilterOutliers`: Enables filtering outliers from the feature.
/// - `Squared`: Enables squaring the feature.
/// - `Mapped`: Enables mapping the feature (a tuple of `MapSelector` that configurationifies how individual rows will be selected for mapping, and `MapOp` which configurationifies what mapping operation will be applied).
///
/// **Feature row filtering tags**:
///
/// - `FilterOutliers`: Enables filtering outlier rows from the feature's column using Tukey's fence method.
///
/// **Automatic feature extraction tags**:
///
/// - `AddExtractedMonth`: Enables the extracted month feature extraction from that feature. The extracted feature will be named `"<feature_name>_month"`.
/// - `AddExtractedTimestamp`: Enables the extracted Unix timestamp feature extraction from that feature. The extracted feature will be named `"<feature_name>_timestamp"`.
/// - `AddLog10`: Enables the extracted base-10 logarithm feature extraction from that feature. The extracted feature will be named `"log10(<feature_name>)"`.
/// - `AddNormalized`: Enables the extracted normalized feature extraction from that feature. The extracted feature will be named `"<feature_name>_normalized"`.
/// - `AddSquared`: Enables the extracted squared feature extraction from that feature. The extracted feature will be named `"<feature_name>^2"`.
///
/// **"Semi-automatic" feature extraction tags**:
///
/// - `AddFeatureExtractedMonth`: Enables and configurationifies the extracted month feature extraction from that feature (a list of `FeatureTags`).
/// - `AddFeatureExtractedTimestamp`: Enables and configurationifies the extracted Unix timestamp feature extraction from that feature (a list of `FeatureTags`).
/// - `AddFeatureLog10`: Enables and configurationifies the extracted base-10 logarithm feature extraction from that feature (a list of `FeatureTags`).
/// - `AddFeatureNormalized`: Enables and configurationifies the extracted normalized feature extraction from that feature (a list of `FeatureTags`).
/// - `AddFeatureSquared`: Enables and configurationifies the extracted squared feature extraction from that feature (a list of `FeatureTags`).
///
/// **Meta tags**:
///
/// - `Not`: Negates the effect of the following tag.
/// - Some others that are internal and should not be used there.
#[derive(Debug)]
pub enum FeatureTags<'a> {
    /// The `Name` tag configurationifies the name of the feature.
    Name(&'a str),
    /// The `Predicted` tag configurationifies that the feature is an output feature.
    Predicted,
    /// The `DateFormat` tag configurationifies the date format to use for date/time features.
    DateFormat(&'a str),
    /// The `OneHotEncod` tag enables conversion to one-hot encoding of the feature.
    OneHotEncode,
    /// The `ToTimestamp` tag enables conversion of the date/time feature to a Unix timestamp.
    ToTimestamp,
    /// The `ExtractMonth` tag enables conversion of the date/time to its month.
    ExtractMonth,
    /// The `Log10` tag enables applying base-10 logarithm to the feature.
    Log10,
    /// The `Normalized` tag enables normalizing the feature.
    Normalized,
    /// The `FilterOutliers` tag enables filtering outliers from the feature using Tukey's fence method.
    FilterOutliers,
    /// The `Squared` tag enables squaring the feature.
    Squared,
    /// The `UsedInModel` tag enables the feature in the model.
    UsedInModel,
    /// The `IsId` tag identifies the feature as an id.
    IsId,
    /// The `AddExtractedMonth` tag enables the extracted month feature extraction from that feature.
    AddExtractedMonth,
    /// The `AddExtractedTimestamp` tag enables the extracted Unix timestamp feature extraction from that feature.
    AddExtractedTimestamp,
    /// The `AddLog10` tag enables the extracted base-10 logarithm feature extraction from that feature.
    AddLog10,
    /// The `AddNormalized` tag enables the extracted normalized feature extraction from that feature.
    AddNormalized,
    /// The `AddSquared` tag enables the extracted squared feature extraction from that feature.
    AddSquared,
    /// The `Mapped` tag enables mapping the feature (a tuple of `MapSelector` that configurationifies how individual rows will be selected for mapping, and `MapOp` which configurationifies what mapping operation will be applied).
    AddFeatureExtractedMonth(&'a [FeatureTags<'a>]),
    /// The `AddFeatureExtractedTimestamp` tag enables and configurationifies the extracted Unix timestamp feature extraction from that feature (a list of `FeatureTags`).
    AddFeatureExtractedTimestamp(&'a [FeatureTags<'a>]),
    /// The `AddFeatureLog10` tag enables and configurationifies the extracted base-10 logarithm feature extraction from that feature (a list of `FeatureTags`).
    AddFeatureLog10(&'a [FeatureTags<'a>]),
    /// The `AddFeatureNormalized` tag enables and configurationifies the extracted normalized feature extraction from that feature (a list of `FeatureTags`).
    AddFeatureNormalized(&'a [FeatureTags<'a>]),
    /// The `AddFeatureSquared` tag enables and configurationifies the extracted squared feature extraction from that feature (a list of `FeatureTags`).
    AddFeatureSquared(&'a [FeatureTags<'a>]),
    /// The `Mapped` tag enables mapping the feature (a tuple of `MapSelector` that configurationifies how individual rows will be selected for mapping, and `MapOp` which configurationifies what mapping operation will be applied).
    Mapped(MapSelector, MapOp),
    /// The `Not` tag negates the effect of the following tag.
    Not(&'a FeatureTags<'a>),
    /// The `RecurseAdded` tag enables the feature tag to be applied to all added features extracted from the feature tag.
    RecurseAdded(&'a FeatureTags<'a>),
    /// The `ExceptFeatures` tag enables the feature tag to be applied to all features except some features.
    ExceptFeatures(&'a FeatureTags<'a>, &'a [&'a str]),
    /// The `OnlyFeatures` tag enables the feature tag to be applied to only some features.
    OnlyFeatures(&'a FeatureTags<'a>, &'a [&'a str]),
}

impl<'a> FeatureTags<'a> {
    /// Applies the feature tag to the feature.
    pub fn apply(&self, feature: &mut Feature) {
        self.apply_bool(feature, true)
    }

    /// Applies the feature tag to all features except some features.
    pub fn except(&'a self, exceptions: &'a [&'a str]) -> FeatureTags<'a> {
        FeatureTags::ExceptFeatures(self, exceptions)
    }

    /// Applies the feature tag to only some features.
    pub fn only(&'a self, features: &'a [&'a str]) -> FeatureTags<'a> {
        FeatureTags::OnlyFeatures(self, features)
    }

    /// Applies the feature tag to all features and their extracted features.
    pub fn incl_added_features(&'a self) -> FeatureTags<'a> {
        FeatureTags::RecurseAdded(self)
    }

    fn apply_bool(&self, feature: &mut Feature, value: bool) {
        match self {
            FeatureTags::Name(name) => feature.name = name.to_string(),
            FeatureTags::Predicted => feature.predicted = value,
            FeatureTags::DateFormat(date_format) => {
                feature.date_format = Some(date_format.to_string())
            }
            FeatureTags::ToTimestamp => feature.to_timestamp = value,
            FeatureTags::ExtractMonth => feature.extract_month = value,
            FeatureTags::Log10 => feature.log10 = value,
            FeatureTags::Normalized => feature.normalized = value,
            FeatureTags::FilterOutliers => feature.filter_outliers = value,
            FeatureTags::Squared => feature.squared = value,
            FeatureTags::OneHotEncode => feature.one_hot_encoded = value,
            FeatureTags::UsedInModel => feature.used_in_model = value,
            FeatureTags::IsId => feature.is_id = value,
            FeatureTags::AddFeatureExtractedMonth(with_extracted_month) => {
                feature.with_extracted_month =
                    Some(Box::new(Feature::from_tags(with_extracted_month)))
            }
            FeatureTags::AddFeatureExtractedTimestamp(with_extracted_timestamp) => {
                feature.with_extracted_timestamp =
                    Some(Box::new(Feature::from_tags(with_extracted_timestamp)))
            }
            FeatureTags::AddFeatureLog10(with_log10) => {
                feature.with_log10 = Some(Box::new(Feature::from_tags(with_log10)))
            }
            FeatureTags::AddFeatureNormalized(with_normalized) => {
                feature.with_normalized = Some(Box::new(Feature::from_tags(with_normalized)))
            }
            FeatureTags::AddFeatureSquared(with_squared) => {
                feature.with_squared = Some(Box::new(Feature::from_tags(with_squared)))
            }
            FeatureTags::Mapped(map_selector, map_op) => {
                feature.mapped = Some((map_selector.clone(), map_op.clone()))
            }
            FeatureTags::Not(feature_tag) => feature_tag.apply_bool(feature, !value),
            FeatureTags::ExceptFeatures(feature_tag, exceptions) => {
                if !exceptions.contains(&feature.name.as_str()) {
                    feature_tag.apply_bool(feature, value)
                }
            }
            FeatureTags::OnlyFeatures(feature_tag, inclusions) => {
                if inclusions.contains(&feature.name.as_str()) {
                    feature_tag.apply_bool(feature, value)
                }
            }
            FeatureTags::AddExtractedMonth => {
                feature.with_extracted_month =
                    Some(Box::new(Feature::from_tags(&[FeatureTags::Name(
                        &format!("{}_month", feature.name),
                    )])))
            }
            FeatureTags::AddExtractedTimestamp => {
                feature.with_extracted_timestamp =
                    Some(Box::new(Feature::from_tags(&[FeatureTags::Name(
                        &format!("{}_timestamp", feature.name),
                    )])))
            }
            FeatureTags::AddLog10 => {
                feature.with_log10 = Some(Box::new(Feature::from_tags(&[FeatureTags::Name(
                    &format!("log10({})", feature.name),
                )])))
            }
            FeatureTags::AddNormalized => {
                feature.with_normalized = Some(Box::new(Feature::from_tags(&[FeatureTags::Name(
                    &format!("{}_normalized", feature.name),
                )])))
            }
            FeatureTags::AddSquared => {
                feature.with_squared = Some(Box::new(Feature::from_tags(&[FeatureTags::Name(
                    &format!("{}^2", feature.name),
                )])))
            }
            FeatureTags::RecurseAdded(feature_tag) => {
                for extracted_feature in feature.get_extracted_features_mut().into_iter() {
                    self.apply_bool(extracted_feature, value)
                }
                feature_tag.apply_bool(feature, value);
            }
        };
    }
}
