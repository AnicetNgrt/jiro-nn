use rand::Rng;
use rand_distr::Distribution;

use super::{MatrixTrait, Scalar};

use faer_core::{Mat, mul::matmul};

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Mat<Scalar>
}

impl MatrixTrait for Matrix {
    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self { data: Mat::with_dims(nrow, ncol, |_, _| 0.0) }
    }

    /// Creates a matrix with random values between min and max (excluded).
    fn random_uniform(nrow: usize, ncol: usize, min: Scalar, max: Scalar) -> Self {
        let mut rng = rand::thread_rng();
        Self { data: Mat::with_dims(nrow, ncol, |_, _| rng.gen_range(min..max)) }
    }

    /// Creates a matrix with random values following a normal distribution.
    fn random_normal(nrow: usize, ncol: usize, mean: Scalar, std_dev: Scalar) -> Self {
        let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
        Self { data: Mat::with_dims(nrow, ncol, |_, _| normal.sample(&mut rand::thread_rng())) }
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
        let vec_data: Vec<Scalar> = data.collect();
        assert_eq!(vec_data.len(), nrow * ncol);
        let vec_data: Vec<Vec<Scalar>> = vec_data.chunks(nrow).map(|slice| slice.to_vec()).collect();
        Self::from_column_leading_matrix(&vec_data)
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
        let data = Mat::with_dims(nrow, ncol, |i, j| m[i][j]);

        Self { data }
    }

    fn from_column_leading_matrix(m: &Vec<Vec<Scalar>>) -> Self {
        let ncol = m.len();
        let nrow = m[0].len();

        let data = Mat::with_dims(nrow, ncol, |i, j| m[j][i]);
        Self { data }
    }

    /// fills a column vector row by row with values of index 0 to v.len()
    fn from_column_vector(v: &Vec<Scalar>) -> Self {
        let ncol = 1;
        let nrow = v.len();

        let data = Mat::with_dims(nrow, ncol, |i, _| v[i]);
        Self { data }
    }

    /// fills a row vector column by column with values of index 0 to v.len()
    fn from_row_vector(v: &Vec<Scalar>) -> Self {
        let ncol = v.len();
        let nrow = 1;

        let data = Mat::with_dims(nrow, ncol, |_, j| v[j]);
        Self { data }
    }

    fn from_fn<F>(nrows: usize, ncols: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> Scalar,
    {
        let data = Mat::with_dims(nrows, ncols, f);
        Self { data }
    }
    
    fn get_column(&self, index: usize) -> Vec<Scalar> {
        let mut result = vec![0.0; self.data.nrows()];
        for i in 0..self.data.nrows() {
            result[i] = self.data.read(i, index);
        }
        result
    }

    fn get_row(&self, index: usize) -> Vec<Scalar> {
        let mut result = vec![0.0; self.data.ncols()];
        for i in 0..self.data.ncols() {
            result[i] = self.data.read(index, i);
        }
        result
    }

    fn columns_map(&self, f: impl Fn(usize, &Vec<Scalar>) -> Vec<Scalar>) -> Self {
        let mut result = vec![vec![0.0; self.data.nrows()]; self.data.ncols()];
        for i in 0..self.data.ncols() {
            let col = self.get_column(i);
            let mapped_col = f(i, &col);
            for j in 0..self.data.nrows() {
                result[i][j] = mapped_col[j];
            }
        }
        Self::from_column_leading_matrix(&result)
    }


    fn map(&self, f: impl Fn(Scalar) -> Scalar + Sync) -> Self {
        let result = Mat::with_dims(self.data.nrows(), self.data.ncols(), |i, j| {
            f(self.data.read(i, j))
        });
        Self { data: result }
    }

    fn map_indexed_mut(&mut self, f: impl Fn(usize, usize, Scalar) -> Scalar + Sync) -> &mut Self {
        let result = Mat::with_dims(self.data.nrows(), self.data.ncols(), |i, j| {
            f(i, j, self.data.read(i, j))
        });
        self.data = result;
        self
    }

    fn dot(&self, other: &Self) -> Self {
        let mut result = Mat::with_dims(self.data.nrows(), other.data.ncols(), |_, _| 0.0);
        matmul(result.as_mut(), self.data.as_ref(), other.data.as_ref(), None, 1.0, faer_core::Parallelism::Rayon(0));
        Self { data: result }
    }

    fn columns_sum(&self) -> Vec<Scalar> {
        let mut result = vec![0.0; self.data.nrows()];
        for j in 0..self.data.ncols() {
            for i in 0..self.data.nrows() {
                result[i] += self.data.read(i, j);
            }
        }
        result
    }

    fn component_mul(&self, other: &Self) -> Self {
        assert_eq!(self.data.nrows(), other.data.nrows());
        assert_eq!(self.data.ncols(), other.data.ncols());
        let result = Mat::with_dims(self.data.nrows(), self.data.ncols(), |i, j| {
            self.data.read(i, j) * other.data.read(i, j)
        });
        Self { data: result }
    }

    fn component_add(&self, other: &Self) -> Self {
        assert_eq!(self.data.nrows(), other.data.nrows());
        assert_eq!(self.data.ncols(), other.data.ncols());
        let result = Mat::with_dims(self.data.nrows(), self.data.ncols(), |i, j| {
            self.data.read(i, j) + other.data.read(i, j)
        });
        Self { data: result }
    }

    fn component_sub(&self, other: &Self) -> Self {
        assert_eq!(self.data.nrows(), other.data.nrows());
        assert_eq!(self.data.ncols(), other.data.ncols());
        let result = Mat::with_dims(self.data.nrows(), self.data.ncols(), |i, j| {
            self.data.read(i, j) - other.data.read(i, j)
        });
        Self { data: result }
    }

    fn component_div(&self, other: &Self) -> Self {
        assert_eq!(self.data.nrows(), other.data.nrows());
        assert_eq!(self.data.ncols(), other.data.ncols());
        let result = Mat::with_dims(self.data.nrows(), self.data.ncols(), |i, j| {
            self.data.read(i, j) / other.data.read(i, j)
        });
        Self { data: result }
    }

    fn transpose(&self) -> Self {
        Self { data: self.data.transpose().to_owned() }
    }

    
    fn dim(&self) -> (usize, usize) {
        (self.data.nrows(), self.data.ncols())
    }

    fn scalar_add(&self, scalar: Scalar) -> Self {
        Self::component_add(&self, &Self { data: Mat::with_dims(self.data.nrows(), self.data.ncols(), |_, _| scalar) })
    }

    fn scalar_mul(&self, scalar: Scalar) -> Self {
        Self::component_mul(&self, &Self { data: Mat::with_dims(self.data.nrows(), self.data.ncols(), |_, _| scalar) })
    }

    fn scalar_sub(&self, scalar: Scalar) -> Self {
        Self::component_sub(&self, &Self { data: Mat::with_dims(self.data.nrows(), self.data.ncols(), |_, _| scalar) })
    }

    fn scalar_div(&self, scalar: Scalar) -> Self {
        Self::component_div(&self, &Self { data: Mat::with_dims(self.data.nrows(), self.data.ncols(), |_, _| scalar) })
    }

    fn index(&self, row: usize, col: usize) -> Scalar {
        self.data.read(row, col)
    }

    fn index_mut(&mut self, row: usize, col: usize) -> &mut Scalar {
        let data = self.data.as_mut();
        data.get(row, col)
    }

    fn get_data(&self) -> Vec<Vec<Scalar>> {
        let mut result = vec![vec![0.0; self.data.nrows()]; self.data.ncols()];
        for i in 0..self.data.nrows() {
            for j in 0..self.data.ncols() {
                result[j][i] = self.data.read(i, j);
            }
        }
        result
    }

    fn get_data_row_leading(&self) -> Vec<Vec<Scalar>> {
        let mut result = vec![vec![0.0; self.data.ncols()]; self.data.nrows()];
        for i in 0..self.data.nrows() {
            for j in 0..self.data.ncols() {
                result[i][j] = self.data.read(i, j);
            }
        }
        result
    }
}
