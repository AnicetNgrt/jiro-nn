use rand::Rng;
use rand_distr::Distribution;

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
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Self { nrow, ncol, data: vec![vec![0.0; nrow]; ncol] }
    }

    /// Creates a matrix with random values between min and max (excluded).
    pub fn random_uniform(nrow: usize, ncol: usize, min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| rng.gen_range(min..max)).collect())
            .collect();
        Self { nrow, ncol, data }
    }

    /// Creates a matrix with random values following a normal distribution.
    pub fn random_normal(nrow: usize, ncol: usize, mean: f64, std_dev: f64) -> Self {
        let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
        let data: Vec<Vec<f64>> = (0..ncol)
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
    pub fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = f64>) -> Self {
        let data: Vec<f64> = data.collect();
        assert_eq!(data.len(), nrow * ncol);
        let data: Vec<Vec<f64>> = data.chunks(nrow).map(|slice| slice.to_vec()).collect();
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
    pub fn from_row_leading_matrix(m: Vec<Vec<f64>>) -> Self {
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

    /// fills a column row by row with values 0 to v.len()
    pub fn from_column_vector(v: Vec<f64>) -> Self {
        Self { nrow: v.len(), ncol: 1, data: vec![v] }
    }

    /// fills a row column by column with values 0 to v.len()
    pub fn from_row_vector(v: Vec<f64>) -> Self {
        let mut result = vec![vec![0.0; v.len()]; 1];
        for (i, col) in v.iter().enumerate() {
            result[0][i] = *col;
        }
        Self { nrow: v.len(), ncol: 1, data: result }
    }

    pub fn get_column(&self, index: usize) -> Vec<f64> {
        self.data[index].clone()
    }

    pub fn get_row(&self, index: usize) -> Vec<f64> {
        let mut result = vec![0.0; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            result[i] = col[index];
        }
        result
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[i][j] = f(*row);
            }
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.ncol, other.nrow);
        let mut result = vec![vec![0.0; other.ncol]; self.nrow];
        
        for i in 0..self.nrow {
            for j in 0..other.ncol {
                for k in 0..self.ncol {
                    result[i][j] += self.data[k][i] * other.data[j][k];
                }
            }
        }

        Self { nrow: other.ncol, ncol: self.nrow, data: result }
    }

    pub fn columns_sum(&self) -> Vec<f64> {
        let mut result = vec![0.0; self.nrow];
        for col in self.data.iter() {
            for (i, val) in col.iter().enumerate() {
                result[i] += val;
            }
        }
        result
    }

    pub fn component_mul(&self, other: &Self) -> Self {
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

    pub fn component_add(&self, other: &Self) -> Self {
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

    pub fn component_sub(&self, other: &Self) -> Self {
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

    pub fn component_div(&self, other: &Self) -> Self {
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

    pub fn transpose(&self) -> Self {
        let mut result = vec![vec![0.0; self.ncol]; self.nrow];
        for (i, col) in self.data.iter().enumerate() {
            for (j, row) in col.iter().enumerate() {
                result[j][i] = *row;
            }
        }
        Self { nrow: self.ncol, ncol: self.nrow, data: result }
    }

    pub fn get_data(&self) -> Vec<Vec<f64>> {
        self.data.clone()
    }

    pub fn get_data_row_leading(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.ncol]; self.nrow];
        for (j, col) in self.data.iter().enumerate() {
            for (i, row) in col.iter().enumerate() {
                result[i][j] = *row;
            }
        }
        result
    }
    
    /// returns the dimensions of the matrix (nrow, ncol)
    pub fn dim(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }
}
