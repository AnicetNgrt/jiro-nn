use serde::{Deserialize, Serialize};
use serde_aux::field_attributes::bool_true;

#[derive(Serialize, Debug, Deserialize, Clone, Hash, Default)]
pub struct Dataset {
    pub name: String,
    pub features: Vec<Feature>,
}

impl Dataset {
    pub fn with_added_feature(&self, feature: Feature) -> Self {
        let mut features = self.features.clone();
        features.push(feature);
        Self {
            name: self.name.clone(),
            features,
        }
    }

    pub fn with_replaced_feature(&self, old_feature_name: &str, feature: Feature) -> Self {
        let mut features = self.features.clone();
        let index = features
            .iter()
            .position(|f| f.name == old_feature_name)
            .unwrap();
        features[index] = feature;
        Self {
            name: self.name.clone(),
            features,
        }
    }

    pub fn feature_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            names.push(feature.name.as_str());
        }
        names
    }

    pub fn in_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if !feature.out && !feature.is_id && !feature.date_format.is_some() {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    pub fn out_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if feature.out {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    pub fn pred_out_features_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for feature in &self.features {
            if feature.out {
                names.push(format!("pred_{}", feature.name));
            }
        }
        names
    }

    pub fn new(name: &str, features: &[Feature]) -> Self {
        Self {
            name: name.to_string(),
            features: features.to_vec(),
        }
    }

    /// The `from_features_options` method is a constructor function for creating a `Dataset` object from a collection of `FeatureOptions`.
    /// 
    /// This method takes in a name for the dataset, as well as a collection of collections of `FeatureOptions` representing the individual features to be included in the dataset. The `FeatureOptions` for each feature specify how that feature should be preprocessed before being included in the dataset.
    /// 
    /// The possible `FeatureOptions` are:
    /// 
    /// - `Name`: The name of the feature (a string).
    /// - `Out`: Whether the feature is the target variable (a boolean).
    /// - `DateFormat`: The format of the date if the feature is a date (a string).
    /// - `ToTimestamp`: Whether to convert the feature to a Unix timestamp if the feature is a date (a boolean).
    /// - `ExtractMonth`: Whether to extract the month from the date if the feature is a date (a boolean).
    /// - `Log10`: Whether to apply the base-10 logarithm transformation to the feature (a boolean).
    /// - `Normalized`: Whether to normalize the feature to have zero mean and unit variance (a boolean).
    /// - `FilterOutliers`: Whether to filter out outliers using the median absolute deviation (a boolean).
    /// - `Squared`: Whether to add a feature representing the squared value of the feature (a boolean).
    /// - `UsedInModel`: Whether the feature is used in the model (a boolean).
    /// - `IsId`: Whether the feature is an identifier (a boolean).
    /// - `WithExtractedMonth`: Enables and specifies the extracted month feature extraction from that feature (a collection of `FeatureOptions`).
    /// - `WithExtractedTimestamp`: Enables and specifies the extracted timestamp feature extraction from that feature (a collection of `FeatureOptions`).
    /// - `WithLog10`: Enables and specifies the extracted base-10 logarithm feature extraction from that feature (a collection of `FeatureOptions`).
    /// - `WithNormalized`: Enables and specifies the normalized feature extraction from that feature (a collection of `FeatureOptions`).
    /// - `WithSquared`: Enables and specifies the extracted squared feature extraction from that feature (a collection of `FeatureOptions`).
    ///
    /// Example:
    /// 
    /// ```
    /// let features1 = &[
    ///     FeatureOptions::Name("age"),
    ///     FeatureOptions::Normalized(true),
    ///     FeatureOptions::Squared(true),
    /// ];
    /// let features2 = &[
    ///     FeatureOptions::Name("income"),
    ///     FeatureOptions::Log10(true),
    ///     FeatureOptions::FilterOutliers(true),
    /// ];
    ///
    /// let dataset = from_features_options("my_dataset", &[&features1, &features2]);
    /// ```

    pub fn from_features_options(name: &str, features: &[&[FeatureOptions]]) -> Self {
        let mut dataset = Self::default();
        dataset.name = name.to_string();
        for feature_options in features {
            let feature = Feature::from_options(feature_options);
            dataset.features.push(feature);
        }
        dataset
    }
}

#[derive(Default, Serialize, Debug, Deserialize, Clone, Hash)]
pub struct Feature {
    pub name: String,
    #[serde(default)]
    pub out: bool,
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
    pub squared: bool,
    pub with_extracted_timestamp: Option<Box<Feature>>,
    pub with_extracted_month: Option<Box<Feature>>,
    pub with_log10: Option<Box<Feature>>,
    pub with_normalized: Option<Box<Feature>>,
    pub with_squared: Option<Box<Feature>>,
    #[serde(default = "bool_true")]
    pub used_in_model: bool,
    #[serde(default)]
    pub is_id: bool,
}

impl Feature {
    /// The `from_options` method is a constructor function for creating a `Feature` object from a list of `FeatureOptions`.
    ///
    /// The possible options are:
    ///
    /// - `Name`: The name of the feature (a string).
    /// - `Out`: Whether or not the feature is an output feature (a boolean).
    /// - `DateFormat`: The date format to use for date/time features (a string).
    /// - `ToTimestamp`: Whether or not to convert the date/time feature to a Unix timestamp (a boolean).
    /// - `ExtractMonth`: Whether or not to extract the month from the date/time feature (a boolean).
    /// - `Log10`: Whether or not to apply base-10 logarithm to the feature (a boolean).
    /// - `Normalized`: Whether or not to normalize the feature (a boolean).
    /// - `FilterOutliers`: Whether or not to filter outliers from the feature (a boolean).
    /// - `Squared`: Whether or not to square the feature (a boolean).
    /// - `UsedInModel`: Whether or not the feature will be used in the model (a boolean).
    /// - `IsId`: Whether or not the feature is an identifier feature (a boolean).
    /// - `WithExtractedMonth`: Enables and specifies the extracted month feature extraction from that feature (a list of `FeatureOptions`).
    /// - `WithExtractedTimestamp`: Enables and specifies the extracted Unix timestamp feature extraction from that feature (a list of `FeatureOptions`).
    /// - `WithLog10`: Enables and specifies the extracted base-10 logarithm feature extraction from that feature (a list of `FeatureOptions`).
    /// - `WithNormalized`: Enables and specifies the extracted normalized feature extraction from that feature (a list of `FeatureOptions`).
    /// - `WithSquared`: Enables and specifies the extracted squared feature extraction from that feature (a list of `FeatureOptions`).

    pub fn from_options(feature_options: &[FeatureOptions]) -> Self {
        let mut feature = Feature::default();
        feature.used_in_model = true;
        for feature_option in feature_options {
            match feature_option {
                FeatureOptions::Name(name) => feature.name = name.to_string(),
                FeatureOptions::Out(out) => feature.out = *out,
                FeatureOptions::DateFormat(date_format) => {
                    feature.date_format = Some(date_format.to_string())
                }
                FeatureOptions::ToTimestamp(to_timestamp) => feature.to_timestamp = *to_timestamp,
                FeatureOptions::ExtractMonth(extract_month) => {
                    feature.extract_month = *extract_month
                }
                FeatureOptions::Log10(log10) => feature.log10 = *log10,
                FeatureOptions::Normalized(normalized) => feature.normalized = *normalized,
                FeatureOptions::FilterOutliers(filter_outliers) => {
                    feature.filter_outliers = *filter_outliers
                }
                FeatureOptions::Squared(squared) => feature.squared = *squared,
                FeatureOptions::UsedInModel(used_in_model) => {
                    feature.used_in_model = *used_in_model
                }
                FeatureOptions::IsId(is_id) => feature.is_id = *is_id,
                FeatureOptions::WithExtractedMonth(with_extracted_month) => {
                    feature.with_extracted_month = Some(Box::new(Feature::from_options(
                        with_extracted_month,
                    )))
                }
                FeatureOptions::WithExtractedTimestamp(with_extracted_timestamp) => {
                    feature.with_extracted_timestamp = Some(Box::new(
                        Feature::from_options(with_extracted_timestamp),
                    ))
                }
                FeatureOptions::WithLog10(with_log10) => {
                    feature.with_log10 = Some(Box::new(Feature::from_options(with_log10)))
                }
                FeatureOptions::WithNormalized(with_normalized) => {
                    feature.with_normalized =
                        Some(Box::new(Feature::from_options(with_normalized)))
                }
                FeatureOptions::WithSquared(with_squared) => {
                    feature.with_squared =
                        Some(Box::new(Feature::from_options(with_squared)))
                }
            }
        }
        feature
    }
}

pub enum FeatureOptions<'a> {
    Name(&'a str),
    Out(bool),
    DateFormat(&'a str),
    ToTimestamp(bool),
    ExtractMonth(bool),
    Log10(bool),
    Normalized(bool),
    FilterOutliers(bool),
    Squared(bool),
    UsedInModel(bool),
    IsId(bool),
    WithExtractedMonth(&'a [FeatureOptions<'a>]),
    WithExtractedTimestamp(&'a [FeatureOptions<'a>]),
    WithLog10(&'a [FeatureOptions<'a>]),
    WithNormalized(&'a [FeatureOptions<'a>]),
    WithSquared(&'a [FeatureOptions<'a>]),
}
