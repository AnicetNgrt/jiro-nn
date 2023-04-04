use crate::loss::mse;

#[derive(Debug)]
pub struct PredStats {
    pub diff: Vec<f64>,
    pub acc: Vec<f64>,
    pub mse: f64,
}

impl PredStats {
    pub fn new(y_pred: Vec<f64>, y_true: Vec<f64>) -> Self {
        let diff: Vec<_> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(y, z)| y - z)
            .collect();
        let acc = diff.iter().map(|d| 1. - d.abs()).collect();
        Self {
            diff,
            acc,
            mse: mse::mse(y_true, y_pred),
        }
    }

    pub fn many_new(y_preds: &Vec<Vec<f64>>, y_trues: &Vec<Vec<f64>>) -> Vec<Self> {
        y_preds.iter()
            .zip(y_trues.iter())
            .map(|(y_pred, y_true)| Self::new(y_pred.to_vec(), y_true.to_vec()))
            .collect()
    }
}

#[derive(Debug)]
pub struct AggregatedPredStats {
    pub min: Option<PredStats>,
    pub max: Option<PredStats>,
    pub avg: Option<PredStats>,
    pub var: Option<PredStats>,
}

fn min_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    vecs.iter().fold(Vec::new(), |acc, row| {
        if acc.is_empty() {
            row.clone()
        } else {
            acc.iter().zip(row.iter()).map(|(a, b)| a.min(*b)).collect()
        }
    })
}

fn max_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    vecs.iter().fold(Vec::new(), |acc, row| {
        if acc.is_empty() {
            row.clone()
        } else {
            acc.iter().zip(row.iter()).map(|(a, b)| a.max(*b)).collect()
        }
    })
}

fn avg_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
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

fn var_vecf64(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    let n_rows = vecs.len();
    let n_cols = vecs[0].len();

    let avg_vec_f64: Vec<f64> = avg_vecf64(vecs);

    // Compute the variance of each column
    let mut var_vec_f64 = vec![0.0; n_cols];
    for row in vecs.iter() {
        for (i, val) in row.iter().enumerate() {
            let diff = val - avg_vec_f64[i];
            var_vec_f64[i] += diff * diff;
        }
    }

    var_vec_f64.iter().map(|x| x / n_rows as f64).collect()
}

fn var_f64(vec: &Vec<f64>) -> f64 {
    let n = vec.len();

    // Compute the average of the vector
    let avg_f64 = vec.iter().sum::<f64>() / n as f64;

    // Compute the variance of the vector
    let mut var_f64 = 0.0;
    for val in vec.iter() {
        let diff = val - avg_f64;
        var_f64 += diff * diff;
    }

    var_f64 / (n - 1) as f64
}

impl AggregatedPredStats {
    pub fn new(stats: Vec<PredStats>, min: bool, max: bool, avg: bool, var: bool) -> Self {
        let diffs: Vec<_> = stats.iter().map(|s| s.diff.clone()).collect();
        let accs: Vec<_> = stats.iter().map(|s| s.acc.clone()).collect();
        let mses: Vec<_> = stats.iter().map(|s| s.mse.clone()).collect();

        let min = if min {
            Some(PredStats {
                diff: min_vecf64(&diffs),
                acc: min_vecf64(&accs),
                mse: *mses.iter().min_by(|a, b| a.total_cmp(b)).unwrap(),
            })
        } else {
            None
        };

        let max = if max {
            Some(PredStats {
                diff: max_vecf64(&diffs),
                acc: max_vecf64(&accs),
                mse: *mses.iter().max_by(|a, b| a.total_cmp(b)).unwrap(),
            })
        } else {
            None
        };

        let avg = if avg {
            Some(PredStats {
                diff: avg_vecf64(&diffs),
                acc: avg_vecf64(&accs),
                mse: mses.iter().sum::<f64>() / mses.len() as f64,
            })
        } else {
            None
        };

        let var = if var {
            Some(PredStats {
                diff: var_vecf64(&diffs),
                acc: var_vecf64(&accs),
                mse: var_f64(&mses),
            })
        } else {
            None
        };

        Self { min, max, avg, var }
    }
}
