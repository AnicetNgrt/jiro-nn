# Backend benchmarking

Compute speed benchmark of the Matrix backends by trying to learn the following function + some Gaussian noise with 1024 params inputs:

```rust
pub fn f(x: &Vec<f64>) -> Vec<f64> {
    let mut res = vec![0.];

    for i in 0..x.len() {
        res[0] *= (x[i]).sin();
    }

    res
}
```

[Results available there](./results/).