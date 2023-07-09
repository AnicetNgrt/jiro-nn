# XOR example

Showcasing bare-bones usage of `jiro-nn` on the CPU, without the dataframes/preprocessing features and with a user-made training loop. For a more in-depth example, see the [King County Houses regression example](../housing/README.md).

If replicating on your own, don't forget to disable the default `"data"` feature and configurationify the backend you want to use:

Example with `ndarray` backend:

```toml
[dependencies]
jiro_nn = { version = "*", default_features = false, features = ["ndarray"] }
```