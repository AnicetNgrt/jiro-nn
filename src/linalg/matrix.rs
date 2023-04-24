use rand::Rng;
use rand_distr::Distribution;

use super::{MatrixTrait, Scalar};

/// Column leading matrix
/// ```txt
/// [
///    [col0: row0 row1 ... rowNrow],
///    [col1: row0 row1 ... rowNrow],
///     ...
///    [colNcol: row0 row1 ... rowNrow],
/// ]
/// ```

#[derive(Debug, Clone)]
pub struct Matrix {
    nrow: usize,
    ncol: usize,
    data: Vec<Vec<Scalar>>,
}

impl MatrixTrait for Matrix {
    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self { nrow, ncol, data: vec![vec![0.0; nrow]; ncol] }
    }

    /// Creates a matrix with random values between min and max (excluded).
    fn random_uniform(nrow: usize, ncol: usize, min: Scalar, max: Scalar) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<Scalar>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| rng.gen_range(min..max)).collect())
            .collect();
        Self { nrow, ncol, data }
    }

    /// Creates a matrix with random values following a normal distribution.
    fn random_normal(nrow: usize, ncol: usize, mean: Scalar, std_dev: Scalar) -> Self {
        let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
        let data: Vec<Vec<Scalar>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| normal.sample(&mut rand::thread_rng())).collect())
            .collect();
        Self { nrow, ncol, data }
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
        let data: Vec<Vec<Scalar>> = data.chunks(nrow).map(|slice| slice.to_vec()).collect();
        Self { nrow, ncol, data }
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
        let mut result = vec![vec![0.0; nrow]; ncol];
        for (i, row) in m.iter().enumerate() {
            for (j, col) in row.iter().enumerate() {
                result[j][i] = *col;
            }
        }
        Self { nrow, ncol, data: result }
    }

    fn from_column_leading_matrix(m: &Vec<Vec<Scalar>>) -> Self {
        let ncol = m.len();
        let nrow = m[0].len();
        Self { nrow, ncol, data: m.clone() }
    }

    /// fills a column vector row by row with values of index 0 to v.len()
    fn from_column_vector(v: &Vec<Scalar>) -> Self {
        Self { nrow: v.len(), ncol: 1, data: vec![v.clone()] }
    }

    /// fills a row vector column by column with values of index 0 to v.len()
    fn from_row_vector(v: &Vec<Scalar>) -> Self {
        let mut result = vec![vec![0.0; v.len()]; 1];
        for (i, col) in v.iter().enumerate() {
            result[0][i] = *col;
        }
        Self { nrow: v.len(), ncol: 1, data: result }
    }

    fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> Scalar,
    {
        let mut result = vec![vec![0.0; nrows]; ncols];
        for i in 0..ncols {
            for j in 0..nrows {
                result[i][j] = f(i, j);
            }
        }
        Self { nrow: nrows, ncol: ncols, data: result }
    }
    
    fn get_column(&self, index: usize) -> Vec<Scalar> {
        self.data[index].clone()
    }

    fn get_row(&self, index: usize) -> Vec<Scalar> {
        let mut result = vec![0.0; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            result[i] = col[index];
        }
        result
    }

    fn columns_map(&self, f: impl Fn(usize, &Vec<Scalar>) -> Vec<Scalar>) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            result[i] = f(i, col);
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }


    fn map(&self, f: impl Fn(Scalar) -> Scalar + Sync) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[i][j] = f(*row);
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn map_indexed_mut(&mut self, f: impl Fn(usize, usize, Scalar) -> Scalar + Sync) -> &mut Self {
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                *self.index_mut(i, j) = f(i, j, self.index(i, j));
            }
        }
        self
    }

    fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.ncol, other.nrow);
        let mut result = Matrix::zeros(self.nrow, other.ncol);
        
        for i in 0..self.nrow {
            for j in 0..other.ncol {
                for k in 0..other.nrow {
                    *result.index_mut(i, j) += self.index(i, k) * other.index(k, j);
                }
            }
        }

        result
    }

    fn columns_sum(&self) -> Vec<Scalar> {
        let mut result = vec![0.0; self.nrow];
        for col in self.data.iter() {
            for (i, val) in col.iter().enumerate() {
                result[i] += val;
            }
        }
        result
    }

    fn component_mul(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                result[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn component_add(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                result[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn component_sub(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                result[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn component_div(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                result[i][j] = self.data[i][j] / other.data[i][j];
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn transpose(&self) -> Self {
        let mut result = vec![vec![0.0; self.ncol]; self.nrow];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[j][i] = *row;
            }
        }
        Self { nrow: self.ncol, ncol: self.nrow, data: result }
    }

    
    fn dim(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn scalar_add(&self, scalar: Scalar) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[i][j] = *row + scalar;
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn scalar_mul(&self, scalar: Scalar) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[i][j] = *row * scalar;
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn scalar_sub(&self, scalar: Scalar) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[i][j] = *row - scalar;
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn scalar_div(&self, scalar: Scalar) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[i][j] = *row / scalar;
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn index(&self, row: usize, col: usize) -> Scalar {
        self.data[col][row]
    }

    fn index_mut(&mut self, row: usize, col: usize) -> &mut Scalar {
        &mut self.data[col][row]
    }

    fn get_data(&self) -> Vec<Vec<Scalar>> {
        self.data.clone()
    }

    fn get_data_row_leading(&self) -> Vec<Vec<Scalar>> {
        let mut result = vec![vec![0.0; self.ncol]; self.nrow];
        for (j, col) in self.data.iter().enumerate() {
            for (i, row) in col.iter().enumerate() {
                result[i][j] = *row;
            }
        }
        result
    }
}