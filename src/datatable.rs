#[allow(unused_imports)]
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use crate::linalg::Scalar;

use polars::prelude::*;

pub type DataFrame = polars::frame::DataFrame;

#[derive(Clone, Debug)]
pub struct DataTable(DataFrame);

pub type Series = polars::series::Series;

impl DataTable {
    pub fn describe(&self) -> Self {
        Self(self.0.describe(None).unwrap())
    }

    pub fn new_empty() -> Self {
        Self(DataFrame::new(Vec::<Series>::new()).unwrap())
    }

    pub fn from_dataframe(dataframe: DataFrame) -> Self {
        Self(dataframe)
    }

    pub fn from_columns(columns: Vec<Series>) -> Self {
        Self(DataFrame::new(columns).unwrap())
    }

    pub fn num_rows(&self) -> usize {
        self.0.height()
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
        let prefix = prefix.map(|s| format!("{}_", s.to_string()));
        let result = self
            .0
            .join(
                table.df(),
                [left_on],
                [right_on],
                JoinType::Inner,
                prefix.clone(),
            )
            .unwrap();

        let mut df = result.clone();
        let names = result.get_column_names();

        if let Some(prefix) = prefix {
            for column_name in names {
                if column_name.ends_with(&prefix) {
                    let new_name = format!("{}{}", prefix, column_name.replace(&prefix, ""));
                    df.rename(column_name, &new_name).unwrap();
                }
            }
        }

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

    pub fn with_column_scalar(&self, name: &str, values: &[Scalar]) -> Self {
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
        let path = path.into();
        let extension = path.extension().unwrap().to_str().unwrap();
        match extension {
            "csv" => Self::from_csv_file(path),
            #[cfg(feature = "ipc")]
            "ipc" => Self::from_ipc_file(path),
            #[cfg(not(feature = "ipc"))]
            "ipc" => panic!("You need to enable the ipc feature to read ipc files"),
            #[cfg(feature = "parquet")]
            "parquet" => Self::from_parquet_file(path),
            #[cfg(not(feature = "parquet"))]
            "parquet" => panic!("You need to enable the parquet feature to read parquet files"),
            _ => panic!("Unsupported file format: {}", extension),
        }
    }

    pub fn columns_names_from_file<P>(path: P) -> Vec<String>
    where
        P: Into<PathBuf>,
    {
        let path = path.into();
        let extension = path.extension().unwrap().to_str().unwrap();
        match extension {
            "csv" => Self::columns_names_from_csv_file(path),
            #[cfg(feature = "ipc")]
            "ipc" => Self::columns_names_from_ipc_file(path),
            #[cfg(not(feature = "ipc"))]
            "ipc" => panic!("You need to enable the ipc feature to read ipc files"),
            #[cfg(feature = "parquet")]
            "parquet" => Self::columns_names_from_parquet_file(path),
            #[cfg(not(feature = "parquet"))]
            "parquet" => panic!("You need to enable the parquet feature to read parquet files"),
            _ => panic!("Unsupported file format: {}", extension),
        }
    }

    pub fn columns_names_from_csv_file<P>(path: P) -> Vec<String>
    where
        P: Into<PathBuf>,
    {
        let data = CsvReader::from_path(path.into())
            .unwrap()
            .with_n_rows(Some(1))
            .finish()
            .unwrap();
        let headers = data.get_column_names();
        headers.iter().map(|s| s.to_string()).collect()
    }

    #[cfg(feature = "ipc")]
    pub fn columns_names_from_ipc_file<P>(path: P) -> Vec<String>
    where
        P: Into<PathBuf>,
    {
        let data = IpcReader::new(File::open(path.into()).unwrap())
            .finish()
            .unwrap();
        let headers = data.get_column_names();
        headers.iter().map(|s| s.to_string()).collect()
    }

    #[cfg(feature = "parquet")]
    pub fn columns_names_from_parquet_file<P>(path: P) -> Vec<String>
    where
        P: Into<PathBuf>,
    {
        let data = ParquetReader::new(File::open(path.into()).unwrap())
            .finish()
            .unwrap();
        let headers = data.get_column_names();
        headers.iter().map(|s| s.to_string()).collect()
    }

    #[cfg(feature = "ipc")]
    pub fn from_ipc_file<P>(path: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self(
            IpcReader::new(File::open(path.into()).unwrap())
                .finish()
                .unwrap(),
        )
    }

    #[cfg(feature = "ipc")]
    pub fn to_ipc_file<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let mut file = std::fs::File::create(path).unwrap();
        IpcWriter::new(&mut file)
            .finish(&mut self.0.clone())
            .unwrap();
    }

    #[cfg(feature = "parquet")]
    pub fn from_parquet_file<P>(path: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self(
            ParquetReader::new(File::open(path.into()).unwrap())
                .finish()
                .unwrap(),
        )
    }

    #[cfg(feature = "parquet")]
    pub fn to_parquet_file<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let mut file = std::fs::File::create(path).unwrap();
        ParquetWriter::new(&mut file)
            .finish(&mut self.0.clone())
            .unwrap();
    }

    pub fn from_csv_file<P>(path: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self(CsvReader::from_path(path).unwrap().finish().unwrap())
    }

    pub fn to_csv_file<P>(&self, path: P)
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

    pub fn map_str_column_to_scalar_column(
        &self,
        column: &str,
        new_column: &str,
        f: impl Fn(&str) -> Scalar,
    ) -> Self {
        let series = self
            .0
            .column(column)
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter()
            .map(|p| p.map(|p| f(p)).unwrap_or_default())
            .collect::<Vec<Scalar>>();

        self.with_column_scalar(new_column, &series)
    }

    pub fn filter_by_scalar_column(&self, column: &str, f: impl Fn(Scalar) -> bool) -> Self {
        let binding = self
            .0
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let series = binding.f64().unwrap();
        let mut mask = BooleanChunked::full("mask", true, series.len());
        mask = mask
            .set_at_idx(
                series
                    .into_iter()
                    .filter(|v| {
                        if let Some(v) = v {
                            !f(*v as Scalar)
                        } else {
                            true
                        }
                    })
                    .enumerate()
                    .map(|(i, _)| i as u32),
                Some(false),
            )
            .unwrap();
        Self(self.0.filter(&mask).unwrap())
    }

    pub fn map_scalar_column(&self, column: &str, f: impl Fn(Scalar) -> Scalar) -> Self {
        let mut edited = self.clone();
        let series = edited
            .0
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let array = series.f64().unwrap();
        let series = array
            .into_iter()
            .map(|p| p.map(|p| f(p as Scalar)))
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

    pub fn split_ratio(&self, ratio: Scalar) -> (Self, Self) {
        let nrows = self.0.shape().0;
        let rows_train = (nrows as Scalar * ratio) as usize;
        let (train, validation) = self.split(rows_train, nrows - rows_train);
        (train, validation)
    }

    pub fn as_scalar_hashmap(&self) -> HashMap<String, Vec<Scalar>> {
        let mut hashmap = HashMap::new();
        for column in self.0.get_columns() {
            if column.dtype().is_numeric() {
                hashmap.insert(
                    column.name().to_string(),
                    self.column_to_vector(column.name()),
                );
            }
        }
        hashmap
    }

    /// Returns a vector of vectors where each vector is a row of the dataset
    pub fn to_vectors(&self) -> Vec<Vec<Scalar>> {
        let mut columns_vec = vec![];
        let columns = self.0.get_column_names();
        for column in columns {
            columns_vec.push(self.column_to_vector(column));
        }

        let mut vectors = vec![vec![0.0; self.0.shape().1]; self.0.shape().0];
        let mut x_id = 0;
        for values in columns_vec {
            let mut t_id = 0;
            for v in values.iter() {
                vectors[t_id][self.0.shape().1 - x_id - 1] = *v;
                t_id += 1;
            }
            x_id += 1;
        }
        vectors
    }

    pub fn from_vectors<S: AsRef<str>>(
        columns_names: &[S],
        columns_vectors: &Vec<Vec<Scalar>>,
    ) -> Self {
        let mut data = Self::new_empty();
        for (i, column) in columns_names.iter().enumerate().rev() {
            let mut values = Vec::new();
            for vector in columns_vectors.iter() {
                values.push(vector[i]);
            }
            data = data.append_column(Series::new(column.as_ref(), &values));
        }
        data
    }

    pub fn to_vectors_with_ids(&self, id_column: &str) -> (Vec<usize>, Vec<Vec<Scalar>>) {
        let ids = self.column_to_ids(id_column);
        let vectors = self.drop_column(id_column).to_vectors();
        (ids, vectors)
    }

    fn series_as_vector(series: &Series) -> Vec<Scalar> {
        let series = series.cast(&DataType::Float64).unwrap();

        series
            .f64()
            .unwrap()
            .into_iter()
            .map(|p| p.unwrap() as Scalar)
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

    pub fn column_to_vector(&self, column: &str) -> Vec<Scalar> {
        Self::series_as_vector(self.0.column(column).unwrap())
    }

    pub fn column_to_ids(&self, column: &str) -> Vec<usize> {
        Self::series_as_ids(self.0.column(column).unwrap())
    }

    pub fn flatten_to_vector(&self) -> Vec<Scalar> {
        self.0.iter().flat_map(Self::series_as_vector).collect()
    }

    pub fn columns_to_vector(&self) -> Vec<Vec<Scalar>> {
        self.0.iter().map(Self::series_as_vector).collect()
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
    ) -> (Vec<Vec<Scalar>>, Vec<Vec<Scalar>>) {
        let df = self.clone().sample(size, true);
        let out_batch = df
            .select_columns(&out_columns.to_vec())
            .transpose()
            .columns_to_vector();
        let in_batch = df
            .drop_columns(&out_columns.to_vec())
            .transpose()
            .columns_to_vector();

        (in_batch, out_batch)
    }

    pub fn min_max_column(&self, column: &str) -> (Scalar, Scalar) {
        let series = self
            .0
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let array = series.f64().unwrap();
        let min = array.min().map(|f| f as Scalar).unwrap_or(Scalar::MAX);
        let max = array.max().map(|f| f as Scalar).unwrap_or(Scalar::MIN);
        (min, max)
    }

    pub fn normalize_column(&self, column: &str, min_max: (Scalar, Scalar)) -> Self {
        let (min, max) = min_max;
        let mut edited = self.clone();
        let series = edited
            .0
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let array = series.f64().unwrap();
        let mut serie: Series = array
            .apply(|v| {
                if (min - max).abs() < 0.0001 && (max - 0.0).abs() < 0.00001 {
                    0.0
                } else {
                    (((v as Scalar) - min) / (max - min)).into()
                }
            })
            .into_series();
        serie.rename(column);
        Self(edited.0.replace(column, serie).unwrap().clone())
    }

    pub fn denormalize_column(&self, column: &str, min_max: (Scalar, Scalar)) -> Self {
        let (min, max) = min_max;
        let mut edited = self.clone();
        let series = edited
            .0
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let array = series.f64().unwrap();
        let mut serie: Series = array
            .apply(|v| ((v as Scalar) * (max - min) + min).into())
            .into_series();
        serie.rename(column);
        Self(edited.0.replace(column, serie).unwrap().clone())
    }

    pub fn column_min_max(&self, column: &str) -> (Scalar, Scalar) {
        let series = self
            .0
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let array = series.f64().unwrap();
        let min = array.min().map(|f| f as Scalar).unwrap_or(Scalar::MAX);
        let max = array.max().map(|f| f as Scalar).unwrap_or(Scalar::MIN);
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
fn test_from_to_vectors() {
    let columns = vec!["a", "b", "c"];
    let vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let data = DataTable::from_vectors(&columns, &vectors);
    let vectors2 = data.to_vectors();
    println!("{:?} {:?}", vectors, vectors2);
    assert_eq!(vectors, vectors2);
}
