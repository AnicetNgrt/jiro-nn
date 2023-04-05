use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use polars::prelude::*;

#[derive(Clone)]
pub struct DataTable(pub DataFrame);

impl DataTable {
    pub fn new_empty() -> Self {
        Self(DataFrame::new(Vec::<Series>::new()).unwrap())
    }

    pub fn from_columns(columns: Vec<Series>) -> Self {
        Self(DataFrame::new(columns).unwrap())
    }

    pub fn apppend(&self, lines: DataTable) -> Self {
        Self(self.0.vstack(&lines.df()).unwrap())
    }

    pub fn with_column_f64(&mut self, name: &str, values: &[f64]) -> Self {
        Self(
            self.0
                .with_column(Series::new(name, values))
                .unwrap()
                .clone(),
        )
    }

    pub fn from_file<P>(path: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self(CsvReader::from_path(path).unwrap().finish().unwrap())
    }

    pub fn to_file<P>(&mut self, path: P)
    where
        P: AsRef<Path>,
    {
        let mut file = std::fs::File::create(path).unwrap();
        CsvWriter::new(&mut file).finish(&mut self.0).unwrap();
    }

    pub fn sample(&mut self, n: Option<usize>, shuffle: bool) -> Self {
        let columns = self
            .0
            .sample_n(n.unwrap_or(self.0.shape().0), false, shuffle, None)
            .unwrap();
        Self(columns)
    }

    pub fn split(&mut self, n_head: usize, n_tail: usize) -> (Self, Self) {
        (
            DataTable(self.0.head(Some(n_head))),
            DataTable(self.0.tail(Some(n_tail))),
        )
    }

    fn series_as_vecf64(series: &Series) -> Vec<f64> {
        series
            .f64()
            .unwrap()
            .into_iter()
            .map(|p| p.unwrap())
            .collect()
    }

    pub fn column_to_vecf64(&self, column: &str) -> Vec<f64> {
        Self::series_as_vecf64(self.0.column(column).unwrap())
    }

    pub fn flatten_to_vecf64(&self) -> Vec<f64> {
        self.0.iter().flat_map(Self::series_as_vecf64).collect()
    }

    pub fn columns_to_vecf64(&self) -> Vec<Vec<f64>> {
        self.0.iter().map(Self::series_as_vecf64).collect()
    }

    pub fn transpose(&self) -> DataTable {
        DataTable(self.0.transpose().unwrap())
    }

    pub fn drop_column(&self, column: &str) -> DataTable {
        DataTable(self.0.drop(column).unwrap())
    }

    pub fn drop_columns(&self, columns: &[&str]) -> DataTable {
        let mut df = self.clone();
        for c in columns {
            df = df.drop_column(c);
        }
        df
    }

    pub fn select_columns(&self, columns: &[&str]) -> DataTable {
        DataTable(self.0.select(columns).unwrap())
    }

    pub fn random_in_out_batch(
        &self,
        out_columns: &[&str],
        size: Option<usize>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let df = self.clone().sample(size, true);
        let out_batch = df
            .select_columns(&out_columns.to_vec())
            .transpose()
            .columns_to_vecf64();
        let in_batch = df
            .drop_columns(&out_columns.to_vec())
            .transpose()
            .columns_to_vecf64();

        (in_batch, out_batch)
    }

    pub fn auto_normalize(&mut self, except_columns: Option<&[&str]>) -> Self {
        let except_columns = HashSet::<&str>::from_iter(
            except_columns
                .map(|ex| ex.to_vec())
                .unwrap_or(vec![])
                .into_iter(),
        );

        let min_max = self.0.get_columns().into_iter().map(|serie| {
            let name = serie.name();
            if !except_columns.contains(name) {
                if let Ok(serie) = serie.cast(&DataType::Float64) {
                    return (name, serie.min::<f64>(), serie.max::<f64>());
                }
            }
            (name, None, None)
        });

        let columns = self
            .0
            .get_columns()
            .into_iter()
            .zip(min_max)
            .map(|(serie, opt_extremas)| match opt_extremas {
                (name, Some(min), Some(max)) => {
                    let mut serie: Series = (serie - min) / (max - min);
                    serie.rename(name);
                    serie.clone()
                }
                _ => serie.clone(),
            })
            .collect();

        Self(DataFrame::new(columns).unwrap())
    }

    pub fn df(self) -> DataFrame {
        self.0
    }
}
