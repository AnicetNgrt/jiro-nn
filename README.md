<p align="center"><img width="150" src="/doc_assets/logo.svg" alt="logo"></p>

<h1 align="center">🐟 jiro-nn</h1>

<p align="center" style="padding-bottom: 10px">Low-friction high-detail Deep Learning framework in Rust</p>


- [Usage](#usage)
  - [Installation \& cargo features](#installation--cargo-features)
  - [Bare-bones XOR example](#bare-bones-xor-example)
  - [Preprocessing + CNNs example](#preprocessing--cnns-example)
- [Features](#features)
  - [Scope and goals](#scope-and-goals)
  - [Backends](#backends)
  - [Precision](#precision)
- [Installing Arrayfire](#installing-arrayfire)

## Usage

### Installation & cargo features

Add this in your project's `Cargo.toml` file, by replacing `<BACKEND>` with the backend you want to use (see [Backends](#backends)):

```toml
[dependencies]
jiro_nn = { 
    version = "*", 
    default-features = false, 
    features = ["<BACKEND>", "data"] # "data" is optional and enables the preprocessing and dataframes features
}
```

### Bare-bones XOR example

Predicting the XOR function with a simple neural network:

```rust
let x = vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 1.0],
];

let y = vec![
    vec![0.0], 
    vec![1.0], 
    vec![1.0], 
    vec![0.0]
];

let network_model = NetworkModelBuilder::new()
    .full_dense(3)
        .tanh()
    .end()
    .full_dense(1)
        .tanh()
    .end()
.build();

let in_size = 2;
let mut network = network_model.to_network(in_size);
let loss = Losses::MSE.to_loss();

for epoch in 0..1000 {
    let error = network.train(epoch, &x, &y, &loss, 1);
    println!("Epoch: {} Error: {}", epoch, error);
}
```

### Preprocessing + CNNs example

MNIST (hand-written digits recognition) workflow example:

```rust
// Step 1: Enrich the features of your data (eg. the "columns") with metadata using a Dataset Specification
// The specification is necessary for guiding further steps (preprocessing, training...)

// Extract features from a spreadsheet to start building a dataset specification
// You could also start blank and add the columns and metadata manually
let mut dataset_spec = Dataset::from_file("dataset/train.csv");
// Now we can add metadata to our features
dataset_spec
    // Flag useless features for removal
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
    .load_data_and_spec("dataset/train.csv", dataset_spec)
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
model_eval.to_json_file("mnist_eval.json");

// Save the weights
let model_params = training.take_model();
model_params.to_json_file("mnist_weights.json");

// Save the predictions alongside the original data
let preds_and_ids = pipeline.revert(&preds_and_ids);
pipeline
    .revert(&data)
    .inner_join(&preds_and_ids, "id", "id", Some("pred"))
    .to_csv_file("mnist_values_and_preds.csv");
```

You can then plot the results using a third-party crate like `gnuplot` *(recommended)*, `plotly` *(also recommended)* or even `plotters`.

*For more in-depth examples, with more configurable workflows spanning many scripts, check out the `examples` folder.*

## Features

Since it is a framework, it is quite opinionated and has a lot of features. But here are the main ones:

NNs (Dense Layers, Full Layers...), CNNs (Dense Layers, Direct Layers, Mean Pooling...), everything batched, SGD, Adam, Momentum, Glorot, many activations (Softmax, Tanh, ReLU...), Learning Rate Scheduling, K-Folds, Split training, cacheable and revertable Pipelines (normalization, feature extraction, outliers filtering, values mapping, one-hot-encoding, log scaling...), loss functions (Binary Cross Entropy, Mean Squared Errors), model specification as code, preprocessing specification as code, performance metrics (R²...), tasks monitoring (progress, logging),  multi-backends (CPU, GPU, see [Backends](#backends)), multi-precision (see [Precision](#precision)).

### Scope and goals

Main goals:

- Implement enough algorithms so that it can fit most use-cases
- Don't stop at NNs, also implement CNNs, RNNs, and whatever we can
- Handle side use-cases that could also be considered "backend" (model building, training, preprocessing)
- Craft opinionated APIs for the core features, and wrappers for useful support libraries if they are not simple enough (DataFrames, Linear Algebra...)
- APIs simplification above rigor and error handling (but no unsafe Rust)
- Make it possible to industrialize and configure workflows (eg. data preprocessing, model building, training, evaluation...)
- Trying not to be 1000x harder than Python and 10x slower than C++ (otherwise what's the point?)

Side/future goals:

- Implement rare but interesting algorithms (Direct layers, Forward Layers...)
- WebAssembly and WebGPU support
- Rust-native GPU backend via wgpu (I can dream, right?)
- Python bindings via PyO3
- Graphic tool for model building

Non-goals:

- Data visualization
- Compliance with other frameworks/standards
- Perfect error handling (don't be ashamed of `unwrap` and `expect`)

### Backends

Switch backends via Cargo features:

- `arrayfire` (CPU/GPU)
    - ✅ Vision available
    - ✅ GPU support
    - 🫤 Slower CPU support
    - 🫤 Hard to install (see [Installing Arrayfire](#installing-arrayfire))
    - 🫤 C++ library, segfaults...
- `ndarray`
    - ✅ Fastest CPU backend
    - ✅ Pure Rust
    - 🫤 Vision not (yet) available
    - 🫤 CPU only
- `nalgebra`
    - ✅ Pure Rust
    - 🫤 Vision not available (and not planned)
    - 🫤 CPU only

### Precision

You can enable precision up to `f64` with the `f64` feature.

Precision below `f32` is not supported (yet).

## Installing Arrayfire

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
jiro_nn = { 
    version = "*", 
    default_features = false, 
    features = ["arrayfire"] 
}
```

If you want to use the CUDA capabilities of Arrayfire on Linux (was tested on Windows 11 WSL2 with Ubuntu and a RTX 3060), [check out this guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation).