use polars::prelude::*;

use crate::{datatable::DataTable, loss::mse, network::Network, stats_utils::*};

#[derive(Debug, Clone)]
pub struct Metric {
    pub prop_dist: Vec<f64>,
    pub dist: Vec<f64>,
    pub acc: Vec<f64>,
    pub mse: f64,
}

impl Metric {
    pub fn new(y_pred: Vec<f64>, y_true: Vec<f64>) -> Self {
        let dist: Vec<_> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(y, z)| (y - z).abs())
            .collect();
        let prop_dist: Vec<_> = y_true
            .iter()
            .zip(dist.iter())
            .map(|(t, d)| d / t)
            .collect();

        let acc = dist.iter().map(|d| 1. - d.abs()).collect();
        Self {
            prop_dist,
            dist,
            acc,
            mse: mse::mse(y_true, y_pred),
        }
    }

    pub fn many_new(y_preds: &Vec<Vec<f64>>, y_trues: &Vec<Vec<f64>>) -> Vec<Self> {
        y_preds
            .iter()
            .zip(y_trues.iter())
            .map(|(y_pred, y_true)| Self::new(y_pred.to_vec(), y_true.to_vec()))
            .collect()
    }

    pub fn to_series(&self, prefix: &str, preds_names: &[&str]) -> Vec<Series> {
        let mut series: Vec<Series> = vec![];
        for (i, acc) in self.acc.iter().enumerate() {
            series.push(Series::new(
                &format!("{}{} pred acc", prefix, preds_names[i]),
                &[*acc],
            ))
        }
        for (i, dist) in self.dist.iter().enumerate() {
            series.push(Series::new(
                &format!("{}{} pred dist", prefix, preds_names[i]),
                &[*dist],
            ))
        }
        for (i, dist) in self.prop_dist.iter().enumerate() {
            series.push(Series::new(
                &format!("{}{} pred prop dist", prefix, preds_names[i]),
                &[*dist],
            ))
        }
        series.push(Series::new(&format!("{}mse", prefix), &[self.mse]));

        series
    }
}

pub struct Benchmark {
    prop_dists: Vec<Vec<f64>>,
    diffs: Vec<Vec<f64>>,
    accs: Vec<Vec<f64>>,
    mses: Vec<f64>,
    pred_stats_aggregation: MetricsAggregates,
}

impl Benchmark {
    pub fn from_test_data(
        network: &mut Network,
        x_test: &Vec<Vec<f64>>,
        y_trues: &Vec<Vec<f64>>
    ) -> Self {
        let preds = network.predict_many(x_test);
        Self::from_preds(&preds, y_trues)
    }

    pub fn from_preds(y_preds: &Vec<Vec<f64>>, y_trues: &Vec<Vec<f64>>) -> Self {
        let stats = Metric::many_new(y_preds, y_trues);
        Self::from_stats(stats)
    }

    pub fn from_stats(stats: Vec<Metric>) -> Self {
        let prop_dists: Vec<_> = stats.iter().map(|s| s.prop_dist.clone()).collect();
        let diffs: Vec<_> = stats.iter().map(|s| s.dist.clone()).collect();
        let accs: Vec<_> = stats.iter().map(|s| s.acc.clone()).collect();
        let mses: Vec<_> = stats.iter().map(|s| s.mse.clone()).collect();

        Self {
            prop_dists,
            diffs,
            accs,
            mses,
            pred_stats_aggregation: MetricsAggregates {
                min: None,
                max: None,
                avg: None,
                var: None,
            },
        }
    }

    pub fn compute_min(&mut self) -> &mut Self {
        self.pred_stats_aggregation.min = Some(Metric {
            prop_dist: min_vecf64(&self.prop_dists),
            dist: min_vecf64(&self.diffs),
            acc: min_vecf64(&self.accs),
            mse: *self.mses.iter().min_by(|a, b| a.total_cmp(b)).unwrap(),
        });
        self
    }

    pub fn compute_max(&mut self) -> &mut Self {
        self.pred_stats_aggregation.max = Some(Metric {
            prop_dist: max_vecf64(&self.prop_dists),
            dist: max_vecf64(&self.diffs),
            acc: max_vecf64(&self.accs),
            mse: *self.mses.iter().max_by(|a, b| a.total_cmp(b)).unwrap(),
        });
        self
    }

    pub fn compute_avg(&mut self) -> &mut Self {
        self.pred_stats_aggregation.avg = Some(Metric {
            prop_dist: avg_vecf64(&self.prop_dists),
            dist: avg_vecf64(&self.diffs),
            acc: avg_vecf64(&self.accs),
            mse: self.mses.iter().sum::<f64>() / self.mses.len() as f64,
        });
        self
    }

    pub fn compute_var(&mut self) -> &mut Self {
        self.pred_stats_aggregation.var = Some(Metric {
            prop_dist: var_vecf64(&self.prop_dists),
            dist: var_vecf64(&self.diffs),
            acc: var_vecf64(&self.accs),
            mse: var_f64(&self.mses),
        });
        self
    }

    pub fn compute_all_metrics_aggregates(&mut self) -> &mut Self {
        self.compute_min()
            .compute_max()
            .compute_avg()
            .compute_var()
    }

    pub fn get_result(&self) -> MetricsAggregates {
        self.pred_stats_aggregation.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MetricsAggregates {
    pub min: Option<Metric>,
    pub max: Option<Metric>,
    pub avg: Option<Metric>,
    pub var: Option<Metric>,
}

impl MetricsAggregates {
    pub fn to_datatable(&self, preds_names: &[&str]) -> DataTable {
        let mut series: Vec<Series> = vec![];
        if let Some(min_stats) = &self.min {
            series.append(&mut min_stats.to_series("min ", preds_names))
        }
        if let Some(max_stats) = &self.max {
            series.append(&mut max_stats.to_series("max ", preds_names))
        }
        if let Some(avg_stats) = &self.avg {
            series.append(&mut avg_stats.to_series("avg ", preds_names))
        }
        if let Some(var_stats) = &self.var {
            series.append(&mut var_stats.to_series("var ", preds_names))
        }

        DataTable::from_columns(series)
    }
}