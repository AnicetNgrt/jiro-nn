pub fn min_vecf64(vec: &Vec<f64>) -> f64 {
    let mut min = vec[0];
    for i in 1..vec.len() {
        if vec[i] < min {
            min = vec[i];
        }
    }
    min
}

pub fn min_vecvecf64(vec: &Vec<Vec<f64>>) -> f64 {
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

pub fn max_vecf64(vec: &Vec<f64>) -> f64 {
    let mut max = vec[0];
    for i in 1..vec.len() {
        if vec[i] > max {
            max = vec[i];
        }
    }
    max
}

pub fn max_vecvecf64(vec: &Vec<Vec<f64>>) -> f64 {
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