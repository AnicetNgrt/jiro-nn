# MNIST

## Workflow

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data) and extract it in the `dataset` folder.
2. Clean the data with `cargo run --bin clean`.
3. (Optional) edit `specify.rs` to change the specification as code.
4. Generate specification files with `cargo run --bin specify -- <spec_name>`.
5. Train the model with `cargo run --bin train -- <spec_name>`.