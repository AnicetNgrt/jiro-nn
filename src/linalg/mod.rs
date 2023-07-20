pub enum Backends {
    ArrayFire,
    Nalgebra,
    Ndarray,
}

#[cfg(feature = "f64")]
pub type Scalar = Scalar;

#[cfg(not(feature = "f64"))]
pub type Scalar = f32;

#[cfg(feature = "arrayfire")]
pub mod arrayfire_matrix;

#[cfg(feature = "arrayfire")]
pub use arrayfire_matrix::Matrix;
#[cfg(feature = "arrayfire")]
pub const BACKEND: Backends = Backends::ArrayFire;

#[cfg(feature = "nalgebra")]
pub mod nalgebra_matrix;

#[cfg(all(feature = "nalgebra", not(feature = "arrayfire")))]
pub use nalgebra_matrix::Matrix;
#[cfg(all(feature = "nalgebra", not(feature = "arrayfire")))]
pub const BACKEND: Backends = Backends::Nalgebra;

#[cfg(feature = "ndarray")]
pub mod ndarray_matrix;

#[cfg(all(
    feature = "ndarray",
    not(feature = "arrayfire"),
    not(feature = "nalgebra")
))]
pub use ndarray_matrix::Matrix;
#[cfg(all(
    feature = "ndarray",
    not(feature = "arrayfire"),
    not(feature = "nalgebra")
))]
pub const BACKEND: Backends = Backends::Ndarray;

pub trait MatrixTrait: Clone {
    fn is_backend_thread_safe() -> bool;

    fn zeros(nrow: usize, ncol: usize) -> Self;

    fn constant(nrow: usize, ncol: usize, value: Scalar) -> Self;

    fn identity(n: usize) -> Self;

    /// Creates a matrix with random values between min and max (excluded).
    fn random_uniform(nrow: usize, ncol: usize, min: Scalar, max: Scalar) -> Self;

    /// Creates a matrix with random values following a normal distribution.
    fn random_normal(nrow: usize, ncol: usize, mean: Scalar, std_dev: Scalar) -> Self;

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
    fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = Scalar>) -> Self;

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
    fn from_row_leading_vector2(m: &Vec<Vec<Scalar>>) -> Self;

    fn from_column_leading_vector2(m: &Vec<Vec<Scalar>>) -> Self;

    /// fills a column vector row by row with values of index 0 to v.len()
    fn from_column_vector(v: &Vec<Scalar>) -> Self;

    /// fills a row vector column by column with values of index 0 to v.len()
    fn from_row_vector(v: &Vec<Scalar>) -> Self;

    fn from_fn<F>(nrows: usize, ncols: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> Scalar;

    fn get_column_as_matrix(&self, idx: usize) -> Self;

    fn from_column_matrices(columns: &[Self]) -> Self;

    fn columns_map(&self, f: impl Fn(usize, &Vec<Scalar>) -> Vec<Scalar>) -> Self;

    fn get_column(&self, index: usize) -> Vec<Scalar>;

    fn get_row(&self, index: usize) -> Vec<Scalar>;

    fn map(&self, f: impl Fn(Scalar) -> Scalar + Sync) -> Self;

    fn map_indexed_mut(&mut self, f: impl Fn(usize, usize, Scalar) -> Scalar + Sync) -> &mut Self;

    fn dot(&self, other: &Self) -> Self;

    fn columns_sum(&self) -> Self;

    fn transpose(&self) -> Self;

    fn get_data_col_leading(&self) -> Vec<Vec<Scalar>>;

    fn get_data_row_leading(&self) -> Vec<Vec<Scalar>>;

    /// returns the dimensions of the matrix (nrow, ncol)
    fn dim(&self) -> (usize, usize);

    fn component_mul(&self, other: &Self) -> Self;

    fn component_add(&self, other: &Self) -> Self;

    fn component_sub(&self, other: &Self) -> Self;

    fn component_div(&self, other: &Self) -> Self;

    fn scalar_add(&self, scalar: Scalar) -> Self;

    fn scalar_mul(&self, scalar: Scalar) -> Self;

    fn scalar_sub(&self, scalar: Scalar) -> Self;

    fn scalar_div(&self, scalar: Scalar) -> Self;

    fn square(&self) -> Self;

    fn sum(&self) -> Scalar;

    fn mean(&self) -> Scalar;

    fn exp(&self) -> Self;

    fn max(&self) -> Scalar;

    fn min(&self) -> Scalar;

    fn maxof(&self, other: &Self) -> Self;

    fn sign(&self) -> Self;

    fn minof(&self, other: &Self) -> Self;

    fn sqrt(&self) -> Self;

    fn log(&self) -> Self;

    fn index(&self, row: usize, col: usize) -> Scalar;

    fn index_mut(&mut self, row: usize, col: usize) -> &mut Scalar;
}
