pub fn min_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    vecs.iter().fold(Vec::new(), |acc, row| {
        if acc.is_empty() {
            row.clone()
        } else {
            acc.iter().zip(row.iter()).map(|(a, b)| a.min(*b)).collect()
        }
    })
}

pub fn max_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    vecs.iter().fold(Vec::new(), |acc, row| {
        if acc.is_empty() {
            row.clone()
        } else {
            acc.iter().zip(row.iter()).map(|(a, b)| a.max(*b)).collect()
        }
    })
}

pub fn avg_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    let n_rows = vecs.len();
    let n_cols = vecs[0].len();

    let mut sum_vec_f64 = vec![0.0; n_cols];

    for row in vecs.iter() {
        for (i, val) in row.iter().enumerate() {
            sum_vec_f64[i] += val;
        }
    }

    sum_vec_f64.iter().map(|x| x / n_rows as f64).collect()
}

pub fn var_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    let n_rows = vecs.len();
    let n_cols = vecs[0].len();

    let avg_vec_f64: Vec<f64> = avg_vecf64(vecs);

    // Compute the variance of each column
    let mut var_vec_f64 = vec![0.0; n_cols];
    for row in vecs.iter() {
        for (i, val) in row.iter().enumerate() {
            let dist = val - avg_vec_f64[i];
            var_vec_f64[i] += dist * dist;
        }
    }

    var_vec_f64.iter().map(|x| x / n_rows as f64).collect()
}

pub fn var_f64(vec: &Vec<f64>) -> f64 {
    let n = vec.len();

    // Compute the average of the vector
    let avg_f64 = vec.iter().sum::<f64>() / n as f64;

    // Compute the variance of the vector
    let mut var_f64 = 0.0;
    for val in vec.iter() {
        let dist = val - avg_f64;
        var_f64 += dist * dist;
    }

    var_f64 / (n - 1) as f64
}
