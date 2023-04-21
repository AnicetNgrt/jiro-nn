use rand::seq::SliceRandom;

pub fn avg_tensor(vec: &Vec<f64>) -> f64 {
    vec.iter().sum::<f64>() / vec.len() as f64
}

pub fn median_tensor(vec: &Vec<f64>) -> f64 {
    if vec.len() == 0 {
        return f64::NAN
    }
    let mut vec = vec.clone();
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = vec.len() / 2;
    if vec.len() % 2 == 0 {
        (vec[mid] + vec[mid - 1]) / 2.0
    } else {
        vec[mid]
    }
}

pub fn quartiles_tensor(vec: &Vec<f64>) -> (f64, f64, f64) {
    let mut vec = vec.clone();
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = vec.len() / 2;
    let q1 = if mid % 2 == 0 {
        (vec[mid / 2] + vec[mid / 2 - 1]) / 2.0
    } else {
        vec[mid / 2]
    };
    let q2 = if vec.len() % 2 == 0 {
        (vec[mid] + vec[mid - 1]) / 2.0
    } else {
        vec[mid]
    };
    let q3 = if mid % 2 == 0 {
        (vec[mid + mid / 2] + vec[mid + mid / 2 - 1]) / 2.0
    } else {
        vec[mid + mid / 2]
    };
    (q1, q2, q3)
}

pub fn tensor_boxplot(vals: &Vec<f64>) -> (f64, f64, f64, f64, f64) {
    let (q1, q2, q3) = quartiles_tensor(vals);
    let iqr = q3 - q1;
    let min = q1 - 1.5 * iqr;
    let max = q3 + 1.5 * iqr;
    (q1, q2, q3, min, max)
}

pub fn min_tensor(vec: &Vec<f64>) -> f64 {
    let mut min = vec[0];
    for i in 1..vec.len() {
        if vec[i] < min {
            min = vec[i];
        }
    }
    min
}

pub fn min_matrix(vec: &Vec<Vec<f64>>) -> f64 {
    let mut min = vec[0][0];
    for i in 0..vec.len() {
        for j in 0..vec[i].len() {
            if vec[i][j] < min {
                min = vec[i][j];
            }
        }
    }
    min
}

pub fn max_tensor(vec: &Vec<f64>) -> f64 {
    let mut max = vec[0];
    for i in 1..vec.len() {
        if vec[i] > max {
            max = vec[i];
        }
    }
    max
}

pub fn max_matrix(vec: &Vec<Vec<f64>>) -> f64 {
    let mut max = vec[0][0];
    for i in 0..vec.len() {
        for j in 0..vec[i].len() {
            if vec[i][j] > max {
                max = vec[i][j];
            }
        }
    }
    max
}

pub fn map_tensor(vec: &Vec<f64>, closure: &dyn Fn(f64) -> f64) -> Vec<f64> {
    vec.iter().map(|x| closure(*x)).collect()
}

pub fn normalize_tensor(vec: &Vec<f64>) -> (Vec<f64>, f64, f64) {
    let min = min_tensor(vec);
    let max = max_tensor(vec);
    let range = max - min;
    (vec.iter().map(|x| (x - min) / range).collect(), min, max)
}

pub fn denormalize_tensor(vec: &Vec<f64>, min: f64, max: f64) -> Vec<f64> {
    let range = max - min;
    vec.iter().map(|x| x * range + min).collect()
}

pub fn normalize_matrix(vec: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, f64, f64) {
    let min = min_matrix(vec);
    let max = max_matrix(vec);
    let range = max - min;
    (vec.iter().map(|x| x.iter().map(|y| (y - min) / range).collect()).collect(), min, max)
}

pub fn denormalize_matrix(vec: &Vec<Vec<f64>>, min: f64, max: f64) -> Vec<Vec<f64>> {
    let range = max - min;
    vec.iter().map(|x| x.iter().map(|y| y * range + min).collect()).collect()
}

pub fn tensors_correlation(vec1: &[f64], vec2: &[f64]) -> Option<f64> {
    if vec1.len() != vec2.len() {
        return None;
    }

    let n = vec1.len() as f64;
    let mean1 = vec1.iter().sum::<f64>() / n;
    let mean2 = vec2.iter().sum::<f64>() / n;

    let mut std_dev1 = 0.0;
    let mut std_dev2 = 0.0;
    let mut cov = 0.0;

    for i in 0..n as usize {
        std_dev1 += (vec1[i] - mean1).powi(2);
        std_dev2 += (vec2[i] - mean2).powi(2);
        cov += (vec1[i] - mean1) * (vec2[i] - mean2);
    }

    std_dev1 = (std_dev1 / (n - 1.0)).sqrt();
    std_dev2 = (std_dev2 / (n - 1.0)).sqrt();

    if std_dev1 == 0.0 || std_dev2 == 0.0 {
        return None;
    }

    Some(cov / ((n - 1.0) * std_dev1 * std_dev2))
}

pub fn tensor_sample(vec: &Vec<f64>, sample_size: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut vec = vec.clone();
    vec.shuffle(&mut rng);
    vec.truncate(sample_size);
    vec
}