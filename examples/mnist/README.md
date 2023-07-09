# MNIST

## Workflow

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data) and extract it in the `dataset` folder.
2. Clean the data with `cargo run --bin clean`.
3. (Optional) edit `configurationify.rs` to change the configuration as code.
4. Generate configuration files with `cargo run --bin configurationify -- <configuration_name>`.
5. Train the model with `cargo run --bin train -- <configuration_name>`.