use crate::linalg::Scalar;

use rand::seq::SliceRandom;

pub fn avg_vector(vec: &Vec<Scalar>) -> Scalar {
    vec.iter().sum::<Scalar>() / vec.len() as Scalar
}

pub fn r2_score(y: &Vec<Scalar>, y_hat: &Vec<Scalar>) -> Scalar {
    assert!(y.len() == y_hat.len());
    
    let y_avg = avg_vector(y);
    let mut ssr = 0.0;
    let mut sst = 0.0;

    for i in 0..y.len() {
        ssr += (y[i] - y_hat[i]).powi(2);
        sst += (y[i] - y_avg).powi(2);
    }

    1.0 - (ssr/sst)
}

pub fn median_vector(vec: &Vec<Scalar>) -> Scalar {
    if vec.len() == 0 {
        return Scalar::NAN
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

pub fn quartiles_vector(vec: &Vec<Scalar>) -> (Scalar, Scalar, Scalar) {
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

pub fn vector_boxplot(vals: &Vec<Scalar>) -> (Scalar, Scalar, Scalar, Scalar, Scalar) {
    let (q1, q2, q3) = quartiles_vector(vals);
    let iqr = q3 - q1;
    let min = q1 - 1.5 * iqr;
    let max = q3 + 1.5 * iqr;
    (q1, q2, q3, min, max)
}

pub fn min_vector(vec: &Vec<Scalar>) -> Scalar {
    let mut min = vec[0];
    for i in 1..vec.len() {
        if vec[i] < min {
            min = vec[i];
        }
    }
    min
}

pub fn min_matrix(vec: &Vec<Vec<Scalar>>) -> Scalar {
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

pub fn max_vector(vec: &Vec<Scalar>) -> Scalar {
    let mut max = vec[0];
    for i in 1..vec.len() {
        if vec[i] > max {
            max = vec[i];
        }
    }
    max
}

pub fn max_matrix(vec: &Vec<Vec<Scalar>>) -> Scalar {
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

pub fn map_vector(vec: &Vec<Scalar>, closure: &dyn Fn(Scalar) -> Scalar) -> Vec<Scalar> {
    vec.iter().map(|x| closure(*x)).collect()
}

pub fn normalize_vector(vec: &Vec<Scalar>) -> (Vec<Scalar>, Scalar, Scalar) {
    let min = min_vector(vec);
    let max = max_vector(vec);
    let range = max - min;
    (vec.iter().map(|x| (x - min) / range).collect(), min, max)
}

pub fn denormalize_vector(vec: &Vec<Scalar>, min: Scalar, max: Scalar) -> Vec<Scalar> {
    let range = max - min;
    vec.iter().map(|x| x * range + min).collect()
}

pub fn normalize_matrix(vec: &Vec<Vec<Scalar>>) -> (Vec<Vec<Scalar>>, Scalar, Scalar) {
    let min = min_matrix(vec);
    let max = max_matrix(vec);
    let range = max - min;
    (vec.iter().map(|x| x.iter().map(|y| (y - min) / range).collect()).collect(), min, max)
}

pub fn denormalize_matrix(vec: &Vec<Vec<Scalar>>, min: Scalar, max: Scalar) -> Vec<Vec<Scalar>> {
    let range = max - min;
    vec.iter().map(|x| x.iter().map(|y| y * range + min).collect()).collect()
}

pub fn vectors_correlation(vec1: &[Scalar], vec2: &[Scalar]) -> Option<Scalar> {
    if vec1.len() != vec2.len() {
        return None;
    }

    let n = vec1.len() as Scalar;
    let mean1 = vec1.iter().sum::<Scalar>() / n;
    let mean2 = vec2.iter().sum::<Scalar>() / n;

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

pub fn vector_sample(vec: &Vec<Scalar>, sample_size: usize) -> Vec<Scalar> {
    let mut rng = rand::thread_rng();
    let mut vec = vec.clone();
    vec.shuffle(&mut rng);
    vec.truncate(sample_size);
    vec
}