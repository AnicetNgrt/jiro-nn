use std::{
    collections::HashMap,
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

    pub fn has_column(&self, name: &str) -> bool {
        self.0.column(name).is_ok()
    }

    pub fn get_columns_names(&self) -> Vec<&str> {
        self.0.get_column_names().to_vec()
    }

    pub fn apppend(&self, lines: &DataTable) -> Self {
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

    pub fn add_column_from(&self, other: &Self, column: &str) -> DataTable {
        let column = other.df().column(column).unwrap().clone();
        self.append_column(column)
    }

    pub fn append_table_as_columns(&self, table: &Self) -> Self {
        let df = table.df().clone();
        let columns = df.get_columns();
        Self(self.0.hstack(columns).unwrap())
    }

    pub fn inner_join(
        &self,
        table: &Self,
        left_on: &str,
        right_on: &str,
        prefix: Option<&str>,
    ) -> Self {
        let columns = table.get_columns();
        let (new_columns, right_on) = if let Some(prefix) = prefix {
            let mut new_columns = Vec::new();
            for mut column in columns {
                let name = column.name();
                if self.has_column(name) {
                    let column = column.rename(&format!("{}_{}", prefix, name));
                    new_columns.push(column.clone());
                } else {
                    new_columns.push(column);
                }
            }
            (new_columns, format!("{}_{}", prefix, right_on))
        } else {
            (columns, right_on.to_string())
        };

        let table = DataFrame::new(new_columns).unwrap();

        let df = self.0
            .join(&table, [left_on], [right_on.as_str()], JoinType::Inner, None)
            .unwrap();
        Self(df)
    }

    pub fn with_autoincrement_id_column(&self, name: &str) -> Self {
        let mut id = 0;
        let mut ids = Vec::new();
        for _ in 0..self.0.shape().0 {
            ids.push(id);
            id += 1;
        }
        Self(
            self.0
                .clone()
                .with_column(Series::new(name, ids))
                .unwrap()
                .clone(),
        )
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

    pub fn split_columns_to_datasets<S: AsRef<str>>(&self, columns: &[S]) -> Vec<Self> {
        let mut datasets = Vec::new();
        for column in columns {
            datasets.push(self.get_column_as_table(column.as_ref()))
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

    pub fn with_column_string(&self, name: &str, values: &[String]) -> Self {
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
        CsvWriter::new(&mut file)
            .finish(&mut self.0.clone())
            .unwrap();
    }

    pub fn shuffle(&self) -> Self {
        self.sample(None, true)
    }

    pub fn sample(&self, n: Option<usize>, shuffle: bool) -> Self {
        let columns = self
            .0
            .sample_n(n.unwrap_or(self.0.shape().0), false, shuffle, None)
            .unwrap();
        Self(columns)
    }

    pub fn split(&self, n_head: usize, n_tail: usize) -> (Self, Self) {
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
        let series = self
            .0
            .column(column)
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter()
            .map(|p| p.map(|p| f(p)).unwrap_or_default())
            .collect::<Vec<f64>>();

        self.with_column_f64(new_column, &series)
    }

    pub fn map_f64_column(&self, column: &str, f: impl Fn(f64) -> f64) -> Self {
        let mut edited = self.clone();
        let series = edited.0.column(column).unwrap().cast(&DataType::Float64).unwrap();
        let array = series.f64().unwrap();
        let series = array.into_iter()
            .map(|p| p.map(|p| f(p)))
            .collect::<Series>();

        Self(edited.0.replace(column, series).unwrap().clone())
    }

    pub fn split_k_folds(&self, k: usize, iter: usize) -> (Self, Self) {
        let nrows = self.0.shape().0;
        let rows_per_fold = nrows / k;
        // extract the validation fold inside the dataset
        let (train, validation) = self.split(rows_per_fold * iter, nrows - rows_per_fold * iter);

        let (validation, additional_train) =
            validation.split(rows_per_fold, nrows - rows_per_fold * (iter + 1));

        let train = train.apppend(&additional_train);

        (train, validation)
    }

    pub fn as_f64_hashmap(&self) -> HashMap<String, Vec<f64>> {
        let columns = self.0.get_column_names();
        let mut hashmap = HashMap::new();
        for column in columns {
            hashmap.insert(column.to_string(), self.column_to_tensor(column));
        }
        hashmap
    }

    pub fn to_tensors(&self) -> Vec<Vec<f64>> {
        let columns_hashmaps = self.as_f64_hashmap();
        let mut tensors = vec![vec![0.0; self.0.shape().1]; self.0.shape().0];
        let mut x_id = 0;
        for (_, values) in columns_hashmaps {
            let mut t_id = 0;
            for v in values.iter() {
                tensors[t_id][x_id] = *v;
                t_id += 1;
            }
            x_id += 1;
        }
        tensors
    }

    pub fn from_tensors<S: AsRef<str>>(columns: &[S], tensors: &Vec<Vec<f64>>) -> Self {
        let mut data = Self::new_empty();
        for (i, column) in columns.iter().enumerate().rev() {
            let mut values = Vec::new();
            for tensor in tensors.iter() {
                values.push(tensor[i]);
            }
            data = data.append_column(Series::new(column.as_ref(), &values));
        }
        data
    }

    pub fn to_tensors_with_ids(&self, id_column: &str) -> (Vec<usize>, Vec<Vec<f64>>) {
        let ids = self.column_to_ids(id_column);
        let tensors = self.drop_column(id_column).to_tensors();
        (ids, tensors)
    }

    fn series_as_tensor(series: &Series) -> Vec<f64> {
        let series = series.cast(&DataType::Float64).unwrap();

        series
            .f64()
            .unwrap()
            .into_iter()
            .map(|p| p.unwrap())
            .collect()
    }

    fn series_as_ids(series: &Series) -> Vec<usize> {
        series
            .i32()
            .unwrap()
            .into_iter()
            .map(|p| p.unwrap() as usize)
            .collect()
    }

    pub fn column_to_tensor(&self, column: &str) -> Vec<f64> {
        Self::series_as_tensor(self.0.column(column).unwrap())
    }

    pub fn column_to_ids(&self, column: &str) -> Vec<usize> {
        Self::series_as_ids(self.0.column(column).unwrap())
    }

    pub fn flatten_to_tensor(&self) -> Vec<f64> {
        self.0.iter().flat_map(Self::series_as_tensor).collect()
    }

    pub fn columns_to_tensor(&self) -> Vec<Vec<f64>> {
        self.0.iter().map(Self::series_as_tensor).collect()
    }

    pub fn transpose(&self) -> DataTable {
        DataTable(self.0.transpose().unwrap())
    }

    pub fn drop_column(&self, column: &str) -> DataTable {
        DataTable(self.0.drop(column).unwrap())
    }

    pub fn drop_columns<S: AsRef<str>>(&self, columns: &[S]) -> DataTable {
        let mut df = self.clone();
        for c in columns {
            df = df.drop_column(c.as_ref());
        }
        df
    }

    pub fn select_columns<S: AsRef<str>>(&self, columns: &[S]) -> DataTable {
        DataTable(self.0.select(columns).unwrap())
    }

    pub fn random_order_in_out<S: AsRef<str>>(&self, out_columns: &[S]) -> (DataTable, DataTable) {
        let df = self.clone().sample(None, true);
        let out_batch = df.select_columns(&out_columns);
        let in_batch = df.drop_columns(&out_columns);
        (in_batch, out_batch)
    }

    pub fn random_in_out_samples(
        &self,
        out_columns: &[&str],
        size: Option<usize>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let df = self.clone().sample(size, true);
        let out_batch = df
            .select_columns(&out_columns.to_vec())
            .transpose()
            .columns_to_tensor();
        let in_batch = df
            .drop_columns(&out_columns.to_vec())
            .transpose()
            .columns_to_tensor();

        (in_batch, out_batch)
    }

    pub fn min_max_column(&self, column: &str) -> (f64, f64) {
        let series = self.0.column(column).unwrap().cast(&DataType::Float64).unwrap();
        let array = series.f64().unwrap();
        let min = array.min().unwrap_or(f64::MAX);
        let max = array.max().unwrap_or(f64::MIN);
        (min, max)
    }

    pub fn normalize_column(&self, column: &str, min_max: (f64, f64)) -> Self {
        let (min, max) = min_max;
        let mut edited = self.clone();
        let series = edited.0.column(column).unwrap().cast(&DataType::Float64).unwrap();
        let array = series.f64().unwrap();
        let mut serie: Series = array.apply(|v| (v - min) / (max - min)).into_series();
        serie.rename(column);
        Self(edited.0.replace(column, serie).unwrap().clone())
    }

    pub fn denormalize_column(&self, column: &str, min_max: (f64, f64)) -> Self {
        let (min, max) = min_max;
        let mut edited = self.clone();
        let series = edited.0.column(column).unwrap().cast(&DataType::Float64).unwrap();
        let array = series.f64().unwrap();
        let mut serie: Series = array.apply(|v| v * (max - min) + min).into_series();
        serie.rename(column);
        Self(edited.0.replace(column, serie).unwrap().clone())
    }

    pub fn column_min_max(&self, column: &str) -> (f64, f64) {
        let series = self.0.column(column).unwrap().cast(&DataType::Float64).unwrap();
        let array = series.f64().unwrap();
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

#[test]
fn test_from_to_tensors() {
    let columns = vec!["a", "b", "c"];
    let tensors = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let data = DataTable::from_tensors(&columns, &tensors);
    let tensors2 = data.to_tensors();
    println!("{:?} {:?}", tensors, tensors2);
    assert_eq!(tensors, tensors2);
}
