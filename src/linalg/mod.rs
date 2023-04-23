#[cfg(feature = "nalgebra")]
pub mod nalgebra_matrix;
#[cfg(feature = "nalgebra")]
pub type Matrix = nalgebra_matrix::Matrix;

#[cfg(feature = "linalg-rayon")]
pub mod rayon_matrix;
#[cfg(all(feature = "linalg-rayon", not(feature = "nalgebra")))]
pub type Matrix = rayon_matrix::Matrix;

#[cfg(feature = "linalg")]
pub mod matrix;

#[cfg(all(feature = "linalg", not(feature = "nalgebra"), not(feature = "linalg-rayon")))]
pub type Matrix = matrix::Matrix;

pub trait MatrixTrait: Clone {
    fn zeros(nrow: usize, ncol: usize) -> Self;

    /// Creates a matrix with random values between min and max (excluded).
    fn random_uniform(nrow: usize, ncol: usize, min: f64, max: f64) -> Self;

    /// Creates a matrix with random values following a normal distribution.
    fn random_normal(nrow: usize, ncol: usize, mean: f64, std_dev: f64) -> Self;

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
    fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = f64>) -> Self;

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
    fn from_row_leading_matrix(m: &Vec<Vec<f64>>) -> Self;

    fn from_column_leading_matrix(m: &Vec<Vec<f64>>) -> Self;

    /// fills a column vector row by row with values of index 0 to v.len()
    fn from_column_vector(v: &Vec<f64>) -> Self;

    /// fills a row vector column by column with values of index 0 to v.len()
    fn from_row_vector(v: &Vec<f64>) -> Self;

    fn from_fn<F>(nrows: usize, ncols: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> f64;

    fn columns_map(&self, f: impl Fn(usize, &Vec<f64>) -> Vec<f64>) -> Self;

    fn get_column(&self, index: usize) -> Vec<f64>;

    fn get_row(&self, index: usize) -> Vec<f64>;

    fn map(&self, f: impl Fn(f64) -> f64 + Sync) -> Self;

    fn map_indexed_mut(&mut self, f: impl Fn(usize, usize, f64) -> f64 + Sync) -> &mut Self;

    fn dot(&self, other: &Self) -> Self;

    fn columns_sum(&self) -> Vec<f64>;

    fn component_mul(&self, other: &Self) -> Self;

    fn component_add(&self, other: &Self) -> Self;

    fn component_sub(&self, other: &Self) -> Self;

    fn component_div(&self, other: &Self) -> Self;

    fn transpose(&self) -> Self;

    fn get_data(&self) -> Vec<Vec<f64>>;

    fn get_data_row_leading(&self) -> Vec<Vec<f64>>;
    
    /// returns the dimensions of the matrix (nrow, ncol)
    fn dim(&self) -> (usize, usize);

    fn scalar_add(&self, scalar: f64) -> Self;

    fn scalar_mul(&self, scalar: f64) -> Self;

    fn scalar_sub(&self, scalar: f64) -> Self;

    fn scalar_div(&self, scalar: f64) -> Self;

    fn index(&self, row: usize, col: usize) -> f64;

    fn index_mut(&mut self, row: usize, col: usize) -> &mut f64;
}