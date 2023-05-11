use std::ops::{Add, Div, Mul, Sub};

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::Distribution;

use super::{MatrixTrait, Scalar};

/// Column leading nalgebra Matrix

#[derive(Debug, Clone)]
pub struct Matrix(DMatrix<Scalar>);

impl MatrixTrait for Matrix {
    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self(DMatrix::zeros(nrow, ncol))
    }

    fn constant(nrow: usize, ncol: usize, value: Scalar) -> Self {
        Self(DMatrix::from_element(nrow, ncol, value))
    }

    /// Creates a matrix with random values between min and max (excluded).
    fn random_uniform(nrow: usize, ncol: usize, min: Scalar, max: Scalar) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<Scalar>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| rng.gen_range(min..max)).collect())
            .collect();

        Self(DMatrix::from_row_slice(nrow, ncol, &data.concat()))
    }

    /// Creates a matrix with random values following a normal distribution.
    fn random_normal(nrow: usize, ncol: usize, mean: Scalar, std_dev: Scalar) -> Self {
        let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
        let data: Vec<Vec<Scalar>> = (0..ncol)
            .map(|_| {
                (0..nrow)
                    .map(|_| normal.sample(&mut rand::thread_rng()))
                    .collect()
            })
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
    fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = Scalar>) -> Self {
        let data: Vec<Scalar> = data.collect();
        assert_eq!(data.len(), nrow * ncol);
        Self(DMatrix::from_column_slice(nrow, ncol, &data))
    }

    fn from_fn<F>(nrows: usize, ncols: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> Scalar,
    {
        Self(DMatrix::from_fn(nrows, ncols, f))
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
    fn from_row_leading_matrix(m: &Vec<Vec<Scalar>>) -> Self {
        let ncol = m[0].len();
        let nrow = m.len();
        Self(DMatrix::from_row_slice(nrow, ncol, &m.concat()))
    }

    fn from_column_leading_matrix(m: &Vec<Vec<Scalar>>) -> Self {
        let ncol = m.len();
        let nrow = m[0].len();
        Self(DMatrix::from_column_slice(nrow, ncol, &m.concat()))
    }

    /// fills a column vector row by row with values of index 0 to v.len()
    fn from_column_vector(v: &Vec<Scalar>) -> Self {
        Self(DMatrix::from_column_slice(v.len(), 1, v))
    }

    /// fills a row vector column by column with values of index 0 to v.len()
    fn from_row_vector(v: &Vec<Scalar>) -> Self {
        Self(DMatrix::from_row_slice(1, v.len(), v))
    }

    fn get_column(&self, index: usize) -> Vec<Scalar> {
        self.0.column(index).iter().map(|x| *x).collect()
    }

    fn get_row(&self, index: usize) -> Vec<Scalar> {
        self.0.row(index).iter().map(|x| *x).collect()
    }

    fn columns_map(&self, f: impl Fn(usize, &Vec<Scalar>) -> Vec<Scalar>) -> Self {
        let mut res = Self::zeros(self.0.nrows(), self.0.ncols());
        for i in 0..self.0.ncols() {
            let col = f(i, &self.get_column(i));
            res.0
                .set_column(i, &DVector::from_column_slice(col.as_slice()));
        }
        res
    }

    fn map_indexed_mut(&mut self, f: impl Fn(usize, usize, Scalar) -> Scalar + Sync) -> &mut Self {
        for i in 0..self.0.nrows() {
            for j in 0..self.0.ncols() {
                *self.index_mut(i, j) = f(i, j, self.index(i, j));
            }
        }
        self
    }

    fn map(&self, f: impl Fn(Scalar) -> Scalar + Sync) -> Self {
        Self(self.0.map(f))
    }

    fn dot(&self, other: &Self) -> Self {
        Self(self.0.clone() * other.0.clone())
    }

    fn columns_sum(&self) -> Self {
        self.dot(&Self::constant(self.dim().1, 1, 1.0))
    }

    fn component_mul(&self, other: &Self) -> Self {
        Self(self.0.component_mul(&other.0))
    }

    fn component_add(&self, other: &Self) -> Self {
        Self(self.0.clone().add(&other.0))
    }

    fn component_sub(&self, other: &Self) -> Self {
        Self(self.0.clone().sub(&other.0))
    }

    fn component_div(&self, other: &Self) -> Self {
        Self(self.0.component_div(&other.0))
    }

    fn transpose(&self) -> Self {
        Self(self.0.transpose())
    }

    fn get_data(&self) -> Vec<Vec<Scalar>> {
        let mut result = vec![vec![0.0; self.0.nrows()]; self.0.ncols()];
        for (j, col) in self.0.column_iter().enumerate() {
            for (i, row) in col.iter().enumerate() {
                result[j][i] = *row;
            }
        }
        result
    }

    fn get_data_row_leading(&self) -> Vec<Vec<Scalar>> {
        let mut result = vec![vec![0.0; self.0.ncols()]; self.0.nrows()];
        for (j, col) in self.0.column_iter().enumerate() {
            for (i, row) in col.iter().enumerate() {
                result[i][j] = *row;
            }
        }
        result
    }

    /// returns the dimensions of the matrix (nrow, ncol)
    fn dim(&self) -> (usize, usize) {
        (self.0.nrows(), self.0.ncols())
    }

    fn scalar_add(&self, scalar: Scalar) -> Self {
        Self(self.0.add_scalar(scalar))
    }

    fn scalar_mul(&self, scalar: Scalar) -> Self {
        Self(self.0.clone().mul(scalar))
    }

    fn scalar_sub(&self, scalar: Scalar) -> Self {
        Self(self.0.add_scalar(-scalar))
    }

    fn scalar_div(&self, scalar: Scalar) -> Self {
        Self(self.0.clone().div(scalar))
    }

    fn index(&self, row: usize, col: usize) -> Scalar {
        self.0[(row, col)]
    }

    fn index_mut(&mut self, row: usize, col: usize) -> &mut Scalar {
        self.0.index_mut((row, col))
    }

    fn square(&self) -> Self {
        Self(self.0.clone().map(|x| x.powi(2)))
    }

    fn sum(&self) -> Scalar {
        self.0.sum()
    }

    fn mean(&self) -> Scalar {
        self.0.mean()
    }

    fn exp(&self) -> Self {
        Self(self.0.clone().map(|x| x.exp()))
    }

    fn maxof(&self, other: &Self) -> Self {
        Self(self.0.clone().map(|x| x.max(other.0[(0, 0)])))
    }

    fn sign(&self) -> Self {
        Self(self.0.clone().map(|x| x.signum()))
    }

    fn minof(&self, other: &Self) -> Self {
        Self(self.0.clone().map(|x| x.min(other.0[(0, 0)])))
    }

    fn sqrt(&self) -> Self {
        Self(self.0.clone().map(|x| x.sqrt()))
    }

    fn log(&self) -> Self {
        Self(self.0.clone().map(|x| x.ln()))
    }
}

impl Matrix {
    pub fn print(&self) {
        println!("{:?}", self.0);
    }
}
