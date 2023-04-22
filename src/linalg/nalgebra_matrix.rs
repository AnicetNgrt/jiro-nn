use std::ops::{Add, Sub};

use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::Distribution;

/// Column leading nalgebra Matrix

#[derive(Debug, Clone)]
pub struct Matrix(DMatrix<f64>);

impl Matrix {
    
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Self(DMatrix::zeros(nrow, ncol))
    }

    /// Creates a matrix with random values between min and max (excluded).
    pub fn random_uniform(nrow: usize, ncol: usize, min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| rng.gen_range(min..max)).collect())
            .collect();
        
        Self(DMatrix::from_row_slice(nrow, ncol, &data.concat()))
    }

    /// Creates a matrix with random values following a normal distribution.
    pub fn random_normal(nrow: usize, ncol: usize, mean: f64, std_dev: f64) -> Self {
        let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
        let data: Vec<Vec<f64>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| normal.sample(&mut rand::thread_rng())).collect())
            .collect();
        Self(DMatrix::from_row_slice(nrow, ncol, &data.concat()))
    }

    /// Fills the matrix with the iterator columns after columns by chunking the data by n_rows.
    /// ```txt
    /// Your data : [[col1: row0 row1 ... rowNrow][col2]...[colNcol]]
    /// Result :
    /// [
    ///    [col0: row0 row1 ... rowNrow],
    ///    [col1: row0 row1 ... rowNrow],
    ///     ...
    ///    [colNcol: row0 row1 ... rowNrow],
    /// ]
    /// ```
    pub fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = f64>) -> Self {
        let data: Vec<f64> = data.collect();
        assert_eq!(data.len(), nrow * ncol);
        Self(DMatrix::from_column_slice(nrow, ncol, &data))
    }

    /// ```txt
    /// Your data :
    /// [
    ///    [row0: col0 col1 ... colNcol],
    ///    [row1: col0 col1 ... colNcol],
    ///     ...
    ///    [rowNrow: col0 col1 ... colNcol],
    /// ]
    /// 
    /// Result :
    /// [
    ///    [col0: row0 row1 ... rowNrow],
    ///    [col1: row0 row1 ... rowNrow],
    ///     ...
    ///    [colNcol: row0 row1 ... rowNrow],
    /// ]
    /// ```
    pub fn from_row_leading_matrix(m: &Vec<Vec<f64>>) -> Self {
        let ncol = m[0].len();
        let nrow = m.len();
        Self(DMatrix::from_row_slice(nrow, ncol, &m.concat()))
    }

    pub fn from_column_leading_matrix(m: &Vec<Vec<f64>>) -> Self {
        let ncol = m.len();
        let nrow = m[0].len();
        Self(DMatrix::from_column_slice(nrow, ncol, &m.concat()))
    }

    /// fills a column row by row with values 0 to v.len()
    pub fn from_column_vector(v: &Vec<f64>) -> Self {
        Self(DMatrix::from_column_slice(v.len(), 1, v))
    }

    /// fills a row column by column with values 0 to v.len()
    pub fn from_row_vector(v: &Vec<f64>) -> Self {
        Self(DMatrix::from_row_slice(1, v.len(), v))
    }

    pub fn get_column(&self, index: usize) -> Vec<f64> {
        self.0.column(index).iter().map(|x| *x).collect()
    }

    pub fn get_row(&self, index: usize) -> Vec<f64> {
        self.0.row(index).iter().map(|x| *x).collect()
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self(self.0.map(f))
    }

    pub fn dot(&self, other: &Self) -> Self {
        Self(self.0.clone() * other.0.clone())
    }

    pub fn columns_sum(&self) -> Vec<f64> {
        self.0.column_sum().iter().map(|x| *x).collect()
    }

    pub fn component_mul(&self, other: &Self) -> Self {
        Self(self.0.component_mul(&other.0))
    }

    pub fn component_add(&self, other: &Self) -> Self {
        Self(self.0.clone().add(&other.0))
    }

    pub fn component_sub(&self, other: &Self) -> Self {
        Self(self.0.clone().sub(&other.0))
    }

    pub fn component_div(&self, other: &Self) -> Self {
        Self(self.0.component_div(&other.0))
    }

    pub fn transpose(&self) -> Self {
        Self(self.0.transpose())
    }

    pub fn get_data(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.0.nrows()]; self.0.ncols()];
        for (j, col) in self.0.column_iter().enumerate() {
            for (i, row) in col.iter().enumerate() {
                result[j][i] = *row;
            }
        }
        result
    }

    pub fn get_data_row_leading(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.0.ncols()]; self.0.nrows()];
        for (j, col) in self.0.column_iter().enumerate() {
            for (i, row) in col.iter().enumerate() {
                result[i][j] = *row;
            }
        }
        result
    }
    
    /// returns the dimensions of the matrix (nrow, ncol)
    pub fn dim(&self) -> (usize, usize) {
        (self.0.nrows(), self.0.ncols())
    }
}
