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