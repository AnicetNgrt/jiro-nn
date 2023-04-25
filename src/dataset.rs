use std::path::{PathBuf};

use serde::{Deserialize, Serialize};
use serde_aux::field_attributes::bool_true;

use crate::{pipelines::map::{MapSelector, MapOp}, datatable::DataTable};

#[derive(Serialize, Debug, Deserialize, Clone, Hash, Default)]
pub struct Dataset {
    pub name: String,
    pub features: Vec<Feature>,
}

impl Dataset {
    /// Create a new dataset with a new feature added.
    pub fn with_added_feature(&self, feature: Feature) -> Self {
        let mut features = self.features.clone();
        features.push(feature);
        Self {
            name: self.name.clone(),
            features,
        }
    }

    /// Create a new dataset with a feature replaced.
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

    /// Return the names of the features that are not specified as outputs.
    pub fn in_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if !feature.out && !feature.is_id && !feature.date_format.is_some() {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    /// Return the names of the features that are specified as outputs.
    pub fn out_features_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for feature in &self.features {
            if feature.out {
                names.push(feature.name.as_str());
            }
        }
        names
    }

    /// Return the names of the features that are specified as outputs with a `"pred"` prefix added.
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

    /// Create a new dataset from a csv file by adding all its columns as default features.
    pub fn from_csv<P: Into<PathBuf>>(path: P) -> Self {
        let path = Into::<PathBuf>::into(path);
        
        let binding = path.clone();
        let file_name = binding
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        let table = DataTable::from_file(path);
        let feature_names = table.get_columns_names();
        let mut features = Vec::new();
        for feature_name in feature_names {
            let feature = Feature::from_options(&[FeatureOptions::Name(feature_name)]);
            features.push(feature);
        }
        Self::new(file_name, &features)
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

    /// The `add_opt_to` method adds a `FeatureOptions` to a specific feature in the dataset.
    pub fn add_opt_to(&mut self, feature_name: &str, option: FeatureOptions) -> &mut Self {
        for feature in &mut self.features {
            if feature.name == feature_name {
                option.apply(feature);
            }
        }
        self
    }

    /// The `add_opt` method adds a `FeatureOptions` to all features in the dataset.
    pub fn add_opt(&mut self, option: FeatureOptions) -> &mut Self {
        for feature in &mut self.features {
            option.apply(feature);
        }
        self
    }

    /// The `from_features_options` method is a constructor function for creating a `Dataset` object from a collection of `FeatureOptions`.
    /// 
    /// This method takes in a name for the dataset, as well as a collection of collections of `FeatureOptions` representing the individual features to be included in the dataset. The `FeatureOptions` for each feature specify how that feature should be preprocessed before being included in the dataset.
    /// 
    /// See the `FeatureOptions` documentation for more information on the available options.
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

    pub fn get_id_column(&self) -> Option<&str> {
        for feature in &self.features {
            if feature.is_id {
                return Some(feature.name.as_str());
            }
        }
        None
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
    pub is_id: bool,
}

impl Feature {
    /// The `from_options` method is a constructor function for creating a `Feature` object from a list of `FeatureOptions`.
    ///
    /// See the documentation for `FeatureOptions` for more information on available options.

    pub fn from_options(feature_options: &[FeatureOptions]) -> Self {
        let mut feature = Feature::default();
        feature.used_in_model = true;
        for feature_option in feature_options {
            feature_option.apply(&mut feature);
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

/// Options to be used to describe a feature and its related preprocessing.
///
/// **Feature description options**:
/// 
/// - `Name`: The name of the feature.
/// - `UsedInModel`: Disables the pruning of the feature at the end of the pipeline. All features are used in the model by default.
/// - `Out`: Sets the feature as an output feature. All features are input features by default.
/// - `IsId`: Identifies the feature as an id. All features are not ids by default.
/// - `DateFormat`: The date format to use for date/time features.
///
/// **Feature replacement/mapping options**:
/// 
/// - `ToTimestamp`: Enables conversion of the date/time feature to a Unix timestamp. Requires the feature to have a `DateFormat` specified.
/// - `ExtractMonth`: Enables conversion of the date/time to its month. Requires the feature to have a `DateFormat` specified.
/// - `Log10`: Enables applying base-10 logarithm to the feature.
/// - `Normalized`: Enables normalizing the feature.
/// - `FilterOutliers`: Enables filtering outliers from the feature.
/// - `Squared`: Enables squaring the feature.
/// - `Mapped`: Enables mapping the feature (a tuple of `MapSelector` that specifies how individual rows will be selected for mapping, and `MapOp` which specifies what mapping operation will be applied).
///
/// **Feature row filtering options**:
/// 
/// - `FilterOutliers`: Enables filtering outlier rows from the feature's column using Tukey's fence method.
/// 
/// **Automatic feature extraction options**:
///
/// - `AddExtractedMonth`: Enables the extracted month feature extraction from that feature. The extracted feature will be named `"<feature_name>_month"`.
/// - `AddExtractedTimestamp`: Enables the extracted Unix timestamp feature extraction from that feature. The extracted feature will be named `"<feature_name>_timestamp"`.
/// - `AddLog10`: Enables the extracted base-10 logarithm feature extraction from that feature. The extracted feature will be named `"log10(<feature_name>)"`.
/// - `AddNormalized`: Enables the extracted normalized feature extraction from that feature. The extracted feature will be named `"<feature_name>_normalized"`.
/// - `AddSquared`: Enables the extracted squared feature extraction from that feature. The extracted feature will be named `"<feature_name>^2"`.
///
/// **"Semi-automatic" feature extraction options**:
///
/// - `AddFeatureExtractedMonth`: Enables and specifies the extracted month feature extraction from that feature (a list of `FeatureOptions`).
/// - `AddFeatureExtractedTimestamp`: Enables and specifies the extracted Unix timestamp feature extraction from that feature (a list of `FeatureOptions`).
/// - `AddFeatureLog10`: Enables and specifies the extracted base-10 logarithm feature extraction from that feature (a list of `FeatureOptions`).
/// - `AddFeatureNormalized`: Enables and specifies the extracted normalized feature extraction from that feature (a list of `FeatureOptions`).
/// - `AddFeatureSquared`: Enables and specifies the extracted squared feature extraction from that feature (a list of `FeatureOptions`).
///
/// **Meta options**:
/// 
/// - `Not`: Negates the effect of the following option.
/// - Some others that are internal and should not be used there.
#[derive(Debug)]
pub enum FeatureOptions<'a> {
    /// The `Name` option specifies the name of the feature.
    Name(&'a str),
    /// The `Out` option specifies that the feature is an output feature.
    Out,
    /// The `DateFormat` option specifies the date format to use for date/time features.
    DateFormat(&'a str),
    /// The `ToTimestamp` option enables conversion of the date/time feature to a Unix timestamp.
    ToTimestamp,
    /// The `ExtractMonth` option enables conversion of the date/time to its month.
    ExtractMonth,
    /// The `Log10` option enables applying base-10 logarithm to the feature.
    Log10,
    /// The `Normalized` option enables normalizing the feature.
    Normalized,
    /// The `FilterOutliers` option enables filtering outliers from the feature using Tukey's fence method.
    FilterOutliers,
    /// The `Squared` option enables squaring the feature.
    Squared,
    /// The `UsedInModel` option enables the feature in the model.
    UsedInModel,
    /// The `IsId` option identifies the feature as an id.
    IsId,
    /// The `AddExtractedMonth` option enables the extracted month feature extraction from that feature.
    AddExtractedMonth,
    /// The `AddExtractedTimestamp` option enables the extracted Unix timestamp feature extraction from that feature.
    AddExtractedTimestamp,
    /// The `AddLog10` option enables the extracted base-10 logarithm feature extraction from that feature.
    AddLog10,
    /// The `AddNormalized` option enables the extracted normalized feature extraction from that feature.
    AddNormalized,
    /// The `AddSquared` option enables the extracted squared feature extraction from that feature.
    AddSquared,
    /// The `Mapped` option enables mapping the feature (a tuple of `MapSelector` that specifies how individual rows will be selected for mapping, and `MapOp` which specifies what mapping operation will be applied).
    AddFeatureExtractedMonth(&'a [FeatureOptions<'a>]),
    /// The `AddFeatureExtractedTimestamp` option enables and specifies the extracted Unix timestamp feature extraction from that feature (a list of `FeatureOptions`).
    AddFeatureExtractedTimestamp(&'a [FeatureOptions<'a>]),
    /// The `AddFeatureLog10` option enables and specifies the extracted base-10 logarithm feature extraction from that feature (a list of `FeatureOptions`).
    AddFeatureLog10(&'a [FeatureOptions<'a>]),
    /// The `AddFeatureNormalized` option enables and specifies the extracted normalized feature extraction from that feature (a list of `FeatureOptions`).
    AddFeatureNormalized(&'a [FeatureOptions<'a>]),
    /// The `AddFeatureSquared` option enables and specifies the extracted squared feature extraction from that feature (a list of `FeatureOptions`).
    AddFeatureSquared(&'a [FeatureOptions<'a>]),
    /// The `Mapped` option enables mapping the feature (a tuple of `MapSelector` that specifies how individual rows will be selected for mapping, and `MapOp` which specifies what mapping operation will be applied).
    Mapped(MapSelector, MapOp),
    /// The `Not` option negates the effect of the following option.
    Not(&'a FeatureOptions<'a>),
    /// The `RecurseAdded` option enables the feature option to be applied to all added features extracted from the feature option.
    RecurseAdded(&'a FeatureOptions<'a>),
    /// The `ExceptFeatures` option enables the feature option to be applied to all features except some features.
    ExceptFeatures(&'a FeatureOptions<'a>, &'a [&'a str]),
    /// The `OnlyFeatures` option enables the feature option to be applied to only some features.
    OnlyFeatures(&'a FeatureOptions<'a>, &'a [&'a str])
}

impl<'a> FeatureOptions<'a> {
    /// Applies the feature option to the feature.
    pub fn apply(&self, feature: &mut Feature) {
        self.apply_bool(feature, true)
    }

    /// Applies the feature option to all features except some features.
    pub fn except(&'a self, exceptions: &'a [&'a str]) -> FeatureOptions<'a> {
        FeatureOptions::ExceptFeatures(self, exceptions)
    }

    /// Applies the feature option to only some features.
    pub fn only(&'a self, features: &'a [&'a str]) -> FeatureOptions<'a> {
        FeatureOptions::OnlyFeatures(self, features)
    }

    /// Applies the feature option to all features and their extracted features.
    pub fn incl_added_features(&'a self) -> FeatureOptions<'a> {
        FeatureOptions::RecurseAdded(self)
    }

    fn apply_bool(&self, feature: &mut Feature, value: bool) {
        println!("Applying {:?} to {:?} with {}", self, feature.name, value);

        match self {
            FeatureOptions::Name(name) => feature.name = name.to_string(),
            FeatureOptions::Out => feature.out = value,
            FeatureOptions::DateFormat(date_format) => {
                feature.date_format = Some(date_format.to_string())
            }
            FeatureOptions::ToTimestamp => feature.to_timestamp = value,
            FeatureOptions::ExtractMonth => {
                feature.extract_month = value
            }
            FeatureOptions::Log10 => feature.log10 = value,
            FeatureOptions::Normalized => feature.normalized = value,
            FeatureOptions::FilterOutliers => {
                feature.filter_outliers = value
            }
            FeatureOptions::Squared => feature.squared = value,
            FeatureOptions::UsedInModel => {
                feature.used_in_model = value
            }
            FeatureOptions::IsId => feature.is_id = value,
            FeatureOptions::AddFeatureExtractedMonth(with_extracted_month) => {
                feature.with_extracted_month = Some(Box::new(Feature::from_options(
                    with_extracted_month,
                )))
            }
            FeatureOptions::AddFeatureExtractedTimestamp(with_extracted_timestamp) => {
                feature.with_extracted_timestamp = Some(Box::new(
                    Feature::from_options(with_extracted_timestamp),
                ))
            }
            FeatureOptions::AddFeatureLog10(with_log10) => {
                feature.with_log10 = Some(Box::new(Feature::from_options(with_log10)))
            }
            FeatureOptions::AddFeatureNormalized(with_normalized) => {
                feature.with_normalized =
                    Some(Box::new(Feature::from_options(with_normalized)))
            }
            FeatureOptions::AddFeatureSquared(with_squared) => {
                feature.with_squared =
                    Some(Box::new(Feature::from_options(with_squared)))
            },
            FeatureOptions::Mapped(map_selector, map_op) => {
                feature.mapped = Some((map_selector.clone(), map_op.clone()))
            }
            FeatureOptions::Not(feature_option) => feature_option.apply_bool(feature, !value),
            FeatureOptions::ExceptFeatures(feature_option, exceptions) => {
                if !exceptions.contains(&feature.name.as_str()) {
                    feature_option.apply_bool(feature, value)
                }
            },
            FeatureOptions::OnlyFeatures(feature_option, inclusions) => {
                if inclusions.contains(&feature.name.as_str()) {
                    feature_option.apply_bool(feature, value)
                }
            },
            FeatureOptions::AddExtractedMonth => {
                feature.with_extracted_month = Some(Box::new(Feature::from_options(
                    &[FeatureOptions::Name(&format!("{}_month", feature.name))]
                )))
            },
            FeatureOptions::AddExtractedTimestamp => {
                feature.with_extracted_timestamp = Some(Box::new(Feature::from_options(
                    &[FeatureOptions::Name(&format!("{}_timestamp", feature.name))]
                )))
            },
            FeatureOptions::AddLog10 => {
                feature.with_log10 = Some(Box::new(Feature::from_options(
                    &[FeatureOptions::Name(&format!("log10({})", feature.name))]
                )))
            },
            FeatureOptions::AddNormalized => {
                feature.with_normalized = Some(Box::new(Feature::from_options(
                    &[FeatureOptions::Name(&format!("{}_normalized", feature.name))]
                )))
            },
            FeatureOptions::AddSquared => {
                feature.with_squared = Some(Box::new(Feature::from_options(
                    &[FeatureOptions::Name(&format!("{}^2", feature.name))]
                )))
            },
            FeatureOptions::RecurseAdded(feature_option) => {
                for extracted_feature in feature.get_extracted_features_mut().into_iter() {
                    self.apply_bool(extracted_feature, value)
                };
                feature_option.apply_bool(feature, value);
            },
        };
    }
}