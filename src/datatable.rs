use std::{
    path::{Path, PathBuf},
};

use polars::prelude::*;

#[derive(Clone, Debug)]
pub struct DataTable(DataFrame);

impl DataTable {
    pub fn new_empty() -> Self {
        Self(DataFrame::new(Vec::<Series>::new()).unwrap())
    }

    pub fn from_dataframe(dataframe: DataFrame) -> Self {
        Self(dataframe)
    }

    pub fn from_columns(columns: Vec<Series>) -> Self {
        Self(DataFrame::new(columns).unwrap())
    }

    pub fn apppend(&self, lines: DataTable) -> Self {
        Self(self.0.vstack(&lines.df()).unwrap())
    }

    pub fn append_columns(&self, columns: Vec<Series>) -> Self {
        Self(self.0.hstack(&columns).unwrap())
    }

    pub fn append_column(&self, column: Series) -> Self {
        Self(self.0.hstack(&[column]).unwrap())
    }

    pub fn append_table_as_column(&self, table: &Self) -> Self {
        let df = table.df().clone();
        let first_column = df.get_column_names()[0];
        let column = df.column(first_column).unwrap().clone();
        Self(self.0.hstack(&[column]).unwrap())
    }

    pub fn get_columns(&self) -> Vec<Series> {
        self.0.get_columns().to_vec()
    }

    pub fn get_column(&self, name: &str) -> Series {
        self.0.column(name).unwrap().clone()
    }

    pub fn get_column_as_table(&self, name: &str) -> Self {
        Self(DataFrame::new(vec![self.0.column(name).unwrap().clone()]).unwrap())
    }

    pub fn split_columns_to_datasets(&self, columns: &[&str]) -> Vec<Self> {
        let mut datasets = Vec::new();
        for column in columns {
            datasets.push(self.get_column_as_table(column))
        }
        datasets
    }

    pub fn with_column_f64(&self, name: &str, values: &[f64]) -> Self {
        Self(
            self.0
                .clone()
                .with_column(Series::new(name, values))
                .unwrap()
                .clone(),
        )
    }

    pub fn with_column_str(&self, name: &str, values: &[&str]) -> Self {
        Self(
            self.0
                .clone()
                .with_column(Series::new(name, values))
                .unwrap()
                .clone(),
        )
    }

    pub fn sort_by_column(&self, column: &str) -> Self {
        Self(self.0.sort(&[column], false).unwrap())
    }

    pub fn from_file<P>(path: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self(CsvReader::from_path(path).unwrap().finish().unwrap())
    }

    pub fn to_file<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let mut file = std::fs::File::create(path).unwrap();
        CsvWriter::new(&mut file).finish(&mut self.0.clone()).unwrap();
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

    pub fn map_str_column_to_f64_column(
        &self,
        column: &str,
        new_column: &str,
        f: impl Fn(&str) -> f64,
    ) -> Self {
        let series = self.0.column(column).unwrap().utf8().unwrap()
            .into_iter()
            .map(|p| p.map(|p| f(p)).unwrap_or_default())
            .collect::<Vec<f64>>();

        self.with_column_f64(new_column, &series)
    }

    pub fn map_f64_column(&self, column: &str, f: impl Fn(f64) -> f64) -> Self {
        let mut edited = self.clone();
        let series = edited.0.column(column).unwrap().f64().unwrap()
            .into_iter()
            .map(|p| p.map(|p| f(p)))
            .collect::<Series>();

        Self(edited.0.replace("price", series).unwrap().clone())
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

    pub fn normalize_column(&self, column: &str) -> Self {
        let mut edited = self.clone();
        let array = edited.0.column(column).unwrap().f64().unwrap();
        let min = array.min().unwrap_or(f64::MAX);
        let max = array.max().unwrap_or(f64::MIN);
        let mut serie: Series = array.apply(|v| (v - min) / (max - min)).into_series();
        serie.rename(column);
        Self(edited.0.replace(column, serie).unwrap().clone())
    }

    pub fn denormalize_column(&self, column: &str, min: f64, max: f64) -> Self {
        let mut edited = self.clone();
        let array = edited.0.column(column).unwrap().f64().unwrap();
        let mut serie: Series = array.apply(|v| v * (max - min) + min).into_series();
        serie.rename(column);
        Self(edited.0.replace(column, serie).unwrap().clone())
    }

    pub fn column_min_max(&self, column: &str) -> (f64, f64) {
        let array = self.0.column(column).unwrap().f64().unwrap();
        let min = array.min().unwrap_or(f64::MAX);
        let max = array.max().unwrap_or(f64::MIN);
        (min, max)
    }

    pub fn rename_column(&self, old: &str, new: &str) -> Self {
        Self(self.clone().0.rename(old, new).unwrap().clone())
    }

    fn df(&self) -> &DataFrame {
        &self.0
    }
}
