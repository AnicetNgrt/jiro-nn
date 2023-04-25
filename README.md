Implementing Neural Networks in Rust from scratch + utils for data manipulation as a newbie to machine learning (I never had used a ML framework before).

This was made for the purpose of my own learning. It is obviously not a good NN library by any means. 

Feel free to give feedback.

## Example

Loading a model specification from JSON, applying a data pipeline on it according to its specification, training it using k-fold cross validation, extracting test & training metrics per folds & per epochs, extracting all the predictions made at last epoch on each fold. Saving it all to disk.

```rs
// Loading a model specification from JSON
let mut model = Model::from_json_file("my_model_spec.json");

// Applying a data pipeline on it according to its specification
let mut pipeline = Pipeline::new();
let (updated_dataset_spec, data) = pipeline
    .add(AttachIds::new("id"))
    .add(ExtractMonths)
    .add(ExtractTimestamps)
    .add(Map::new())
    .add(LogScale10::new())
    .add(Square::new())
    .add(FilterOutliers)
    .add(Normalize::new())
    .run("./caching_dir", &model.dataset);

let model = model.with_new_dataset(updated_dataset_spec);

// Training it using k-fold cross validation
// + extracting test & training metrics per folds & per epochs
// + extracting all predictions made during final epoch
let kfold = model.trainer.maybe_kfold().expect("We only do k-folds here!");
let (validation_preds, model_eval) = kfold
    .attach_real_time_reporter(|report| println!("Perf report: {:#?}", report))
    .run(&model, &data);

// Reverting the pipeline on the predictions & data to get interpretable values
let validation_preds = pipeline.revert_columnswise(&validation_preds);
let data = pipeline.revert_columnswise(&data);

// Joining the data and the predictions together
let data_and_preds = data.inner_join(&validation_preds, "id", "id", Some("pred"));

// Saving it all to disk
data_and_preds.to_file(format!("models_stats/{}.csv", config_name));
model_eval.to_json_file(format!("models_stats/{}.json", config_name));
```

## Working features

### Neural Networks

- Neural Network abstraction
- Layers
    - Shared abstraction
        - Forward/Backward passes
        - Handle (mini)batches
    - Dense layers
        - Weights optimizer/initializer (no regularization yet)
        - Biases optimizer/initializer (no regularization yet)
    - Activation layers
        - Linear
        - Hyperbolic Tangent
        - ReLU
        - Sigmoid
        - Tanh
    - Full layers (Dense + Activation)
        - Optional Dropout regularization
- Stochastic Gradient Descent (SGD)
    - Optional optimizers
        - Momentum
        - Adam
    - Learning rate scheduling
        - Constant
        - Piecewise constant
        - Inverse time decay
- Initializers
    - Zeros
    - Random uniform (signed & unsigned)
    - Glorot uniform
- Losses
    - MSE

### Models utils

- Abstractions over training
    - KFolds + model benchmarking in a few LoC
- Model specification
    - Simple API to create as code
    - From/to JSON conversion
    - Specify training parameters (k-folds, epochs...)
    - Specify the dataset
        - Features in/out
        - Features preprocessing
- Model benchmarking
    - Handy utilities to store & compute training metrics

### Preprocessing

- Layer-based data pipelines abstraction
    - Cached & revertible feature mapping/feature extraction layers
        - Normalize
        - Square
        - Log10 scale
        - Extract month from Date string
        - Extract unix timestamp from Date string
        - Limited row mapping (for advanced manipulations)
    - Row filtering layers
        - Filter outliers
    - Other layers
        - Attach IDs column

### Data analysis

- Simple Polars DataFrame wrapper (DataTable)
    - Load & manipulate CSV data
    - Generate input/output vectors
        - Sample
        - Shuffle
        - Split
        - K-folds
- `Vector<f64>` statistics & manipulation utils
    - Stats (mean, std dev, correlation, quartiles...)
    - Normalization

### Linear algebra

- Many backends for the Matrix type (toggled using compile-time cargo features)
    - Feature `nalgebra` (default)
        - Fast
        - CPU-bound
    - Feature `linalg`
        - 100% custom Vec-based
        - Slow
        - CPU-bound
    - Feature `linalg-rayon`
        - linalg parallelized with rayon
        - Way faster than linalg but slower than nalgebra
        - CPU-bound
    - Feature `faer`
        - WIP
        - Should be fast but for now it's on par with linalg-rayon
        - CPU-bound
- Toggling f64 with the `f64` feature (default is f32)

## Examples

### [King County House price regression](./examples/housing/)

Standard looking results by doing roughly [the same approach as a user named frederico on Kaggle using Pytorch](https://www.kaggle.com/code/chavesfm/dnn-house-price-r-0-88/notebook). Involving data manipulation with pipelines, and a 8 layers of ~20 inputs each model using ReLU & Adam. Training over 300 epochs with 8 folds k-folds.

Charts made with the gnuplot crate.

![loss according to training epochs](examples/visuals/full_lt_8ReLU-Adam-Lin-Adam_loss.png)

![prices according to predicted prices](./examples/visuals/full_lt_8ReLU-Adam-Lin-Adam_price.png)

![prices & predicted prices according to lat & long](examples/visuals/full_lt_8ReLU-Adam-Lin-Adam_latlong.png)


### [XOR function (deprecated example)](./examples/xor%5Bdeprecated%5D/)

Standard looking results. 

Charts made with the plotters crate.

![dropout rates](examples/visuals/dropout_rate.png)

![learning rate decays](examples/visuals/learning_rate_decay.png)

![dropout rates](examples/visuals/xor-example-predictions.png)

### [Backend benchmarking](./examples/benchmark/)

Compute speed benchmark of the Matrix backends by trying to learn the following function + some Gaussian noise with 1024 params inputs:

```rs
pub fn f(x: &Vec<f64>) -> Vec<f64> {
    let mut res = vec![0.];

    for i in 0..x.len() {
        res[0] *= (x[i]).sin();
    }

    res
}
```

[Results available there](./examples/benchmark/results/).