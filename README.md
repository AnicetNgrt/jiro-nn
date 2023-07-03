# Neural Networks in Rust

Implementing GPU-bound Neural Networks in Rust from scratch + utils for data manipulation.

This was made for the purpose of my own learning. It is obviously not a production-ready library by any means. 

Feel free to give feedback.

## Preprocessing + CNNs example

MNIST (hand-written digits recognition) workflow example:

```rust
// Step 1: Enrich the features of your data (eg. the "columns") with metadata using Tags
// The tags are necessary for guiding further steps (preprocessing, training...)

// Extract columns from a CSV (or parquet, ipc...) file to start building a dataset specification
// You could also start blank and add the columns and metadata manually
let mut dataset_spec = Dataset::from_file("dataset/train.csv");
// Now we refine it
dataset_spec
    // Remove useless features
    .remove_features(&["size"])
    // Tell the framework which column is an ID (so it can be ignored in training, used in joins, and so on)
    .tag_feature("id", IsId)
    // Tell the framework which column is the feature to predict
    // You could very well declare multiple features as Predicted
    .tag_feature("label", Predicted)
    // Since it is a classification problem, indicate the label needs One-Hot encoding during preprocessing
    .tag_feature("label", OneHotEncode)
    // You may also want to normalize everything except the ID & label during preprocessing
    .tag_all(Normalized.except(&["id", "label"]));

// Step 2: Preprocess the data

// Create a pipeline with all the necessary steps
let mut pipeline = Pipeline::basic_single_pass();
// Run it on the data
let (dataset_spec, data) = pipeline
    .load_data_and_spec("./dataset/train_cleaned.parquet", dataset_spec)
    .run();

// Step 3: Specify and build your model

// A model is tied to a dataset specification
let model = ModelBuilder::new(dataset_spec)
    // Some configuration is also tied to the model
    // All the configuration calls are optional, defaults are picked otherwise
    .batch_size(128)
    .loss(Losses::BCE)
    .epochs(20)
    // Then you can start building the neural network
    .neural_network()
        // Specify all your layers
        // A convolution network is considered a layer of a neural network in this framework
        .conv_network(1)
            // Now the convolution layers
            .full_dense(32, 5)
                // You can set the activation function for any layer and many other parameters
                // Otherwise defaults are picked
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
        // Now we go back to configuring the top-level neural network
        .full_dense(128)
            .relu()
            .adam()
        .end()
        .full_dense(10)
            .softmax()
            .adam()
        .end()
    .end()
    .build();

println!(
    "Model parameters count: {}",
    model.to_network().get_params().count()
);

// Step 4: Train the model

// Monitor the progress of the training on a nice TUI (with other options coming soon)
TM::start_monitoring();
// Use a SplitTraining to split the data into a training and validation set (k-fold also available)
let mut training = SplitTraining::new(0.8);
let (preds_and_ids, model_eval) = training.run(&model, &data);
TM::stop_monitoring();

// Step 5: Save the resulting predictions, weights and model evaluation

// Save the model evaluation per epoch
model_eval.to_json_file(format!("models_stats/{}.json", config_name));

// Save the weights
let model_params = training.take_model();
model_params.to_json(format!("models_stats/{}_params.json", config_name));

// Save the predictions alongside the original data
let preds_and_ids = pipeline.revert(&preds_and_ids);
pipeline
    .revert(&data)
    .inner_join(&preds_and_ids, "id", "id", Some("pred"))
    .to_csv_file("mnist_values_and_preds.parquet");
```

You can then plot the results using a third-party crate like `gnuplot` *(recommended)*, `plotly` *(also recommended)* or even `plotters`.

*For more in-depth examples, with more configurable workflows spanning many scripts, check out the `examples` folder.*

## Include

Add this in your project's `Cargo.toml` file:

```toml
[dependencies]
neural_networks_rust = "*"
```

### Using Arrayfire

You need to first [install Arrayfire](https://arrayfire.org/docs/installing.htm#gsc.tab=0) in order to use the `arrayfire` feature for fast compute on the CPU or the GPU using Arrayfire's C++/CUDA/OpenCL backends (it will first try OpenCL if installed, then CUDA, then C++). Make sure all the steps of the installation work 100% with no weird warning, as they may fail in quite subtle ways.

Once you installed Arrayfire, you:

1. Set the `AF_PATH` to your Arrayfire installation directory (for example: `/opt/Arrayfire`).
2. Add the path to lib files to the environement variables:
    - Linux: `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib64`
    - OSX: `DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$AF_PATH/lib`
    - Windows: Add `%AF_PATH%\lib` to PATH
3. Run `sudo ldconfig` if on Linux
4. Run `cargo clean`
5. Disable default features and activate the `arrayfire` feature

```toml
[dependencies]
neural_networks_rust = { 
    version = "*", 
    default_features = false, 
    features = ["arrayfire"] 
}
```

If you want to use the CUDA capabilities of Arrayfire on Linux (was tested on Windows 11 WSL2 with Ubuntu and a RTX 3060), [check out this guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation).

If you install none of the above, the default `nalgebra` feature is still available for a pure Rust CPU-bound backend.
