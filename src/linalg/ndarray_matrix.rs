use std::fmt;

use ndarray::Array2;
use rand::Rng;

use super::{MatrixTrait, Scalar};

#[derive(Clone, Debug)]
pub struct Matrix(pub Array2<Scalar>);

impl MatrixTrait for Matrix {
    fn is_backend_thread_safe() -> bool {
        true
    }

    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self(Array2::zeros((nrow, ncol)))
    }

    fn constant(nrow: usize, ncol: usize, value: Scalar) -> Self {
        Self(Array2::from_elem((nrow, ncol), value))
    }

    fn identity(n: usize) -> Self {
        let id = Array2::eye(n);
        Self(id)
    }

    fn random_uniform(nrow: usize, ncol: usize, min: Scalar, max: Scalar) -> Self {
        let mat = Array2::from_shape_fn((nrow, ncol), |(_, _)| {
            rand::thread_rng().gen_range(min..max)
        });
        Self(mat)
    }

    fn random_normal(nrow: usize, ncol: usize, mean: Scalar, std_dev: Scalar) -> Self {
        let mat = Array2::from_shape_fn((nrow, ncol), |(_, _)| {
            rand::thread_rng().gen_range((mean - 3. * std_dev)..(mean + 3. * std_dev))
        });
        Self(mat)
    }

    fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = Scalar>) -> Self {
        let data: Vec<Scalar> = data.collect();
        assert_eq!(data.len(), nrow * ncol);
        let mut mat = Array2::<Scalar>::zeros((ncol, nrow));

        for i in 0..nrow * ncol {
            let row = i / ncol;
            let col = i % ncol;
            mat[[col, row]] = data[i];
        }

        Self(mat)
    }

    fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> Scalar,
    {
        let mat = Array2::from_shape_fn((nrows, ncols), |(i, j)| f(i, j));
        Self(mat)
    }

    fn from_row_leading_vector2(m: &Vec<Vec<Scalar>>) -> Self {
        let mat = Array2::from_shape_fn((m.len(), m[0].len()), |(i, j)| m[i][j]);
        Self(mat)
    }

    fn from_column_leading_vector2(m: &Vec<Vec<Scalar>>) -> Self {
        let mat = Array2::from_shape_fn((m[0].len(), m.len()), |(i, j)| m[j][i]);
        Self(mat)
    }

    fn from_column_vector(v: &Vec<Scalar>) -> Self {
        let mat = Array2::from_shape_fn((v.len(), 1), |(i, _)| v[i]);
        Self(mat)
    }

    fn from_row_vector(v: &Vec<Scalar>) -> Self {
        let mat = Array2::from_shape_fn((1, v.len()), |(_, j)| v[j]);
        Self(mat)
    }

    fn from_column_matrices(columns: &[Self]) -> Self {
        let mut mat = Array2::<Scalar>::zeros((columns[0].0.nrows(), columns.len()));

        for i in 0..columns.len() {
            mat.column_mut(i).assign(&columns[i].0);
        }

        Self(mat)
    }

    fn get_column_as_matrix(&self, idx: usize) -> Self {
        let col = self.0.column(idx);
        Self(Array2::from_shape_vec((col.len(), 1), col.to_vec()).unwrap())
    }

    fn get_column(&self, idx: usize) -> Vec<Scalar> {
        let col = self.0.column(idx);
        col.to_vec()
    }

    fn get_row(&self, idx: usize) -> Vec<Scalar> {
        let row = self.0.row(idx);
        row.to_vec()
    }

    fn columns_map(&self, _f: impl Fn(usize, &Vec<Scalar>) -> Vec<Scalar>) -> Self {
        unimplemented!("Incompatible")
    }

    fn map_indexed_mut(&mut self, _f: impl Fn(usize, usize, Scalar) -> Scalar + Sync) -> &mut Self {
        unimplemented!("Incompatible")
    }

    fn map(&self, _f: impl Fn(Scalar) -> Scalar + Sync) -> Self {
        unimplemented!("Incompatible")
    }

    fn dot(&self, other: &Self) -> Self {
        let mat = self.0.dot(&other.0);
        Self(mat)
    }

    fn columns_sum(&self) -> Self {
        self.dot(&Self::constant(self.dim().1, 1, 1.0))
    }

    fn component_mul(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    fn component_add(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }

    fn component_sub(&self, other: &Self) -> Self {
        Self(&self.0 - &other.0)
    }

    fn component_div(&self, other: &Self) -> Self {
        Self(&self.0 / &other.0)
    }

    fn transpose(&self) -> Self {
        let mat = self.0.clone().reversed_axes();
        Self(mat)
    }

    fn get_data_col_leading(&self) -> Vec<Vec<Scalar>> {
        let mut res = vec![];
        for i in 0..self.0.ncols() {
            res.push(self.get_column(i));
        }
        res
    }

    fn get_data_row_leading(&self) -> Vec<Vec<Scalar>> {
        let mut res = vec![];
        for i in 0..self.0.nrows() {
            res.push(self.get_row(i));
        }
        res
    }

    fn dim(&self) -> (usize, usize) {
        (self.0.nrows(), self.0.ncols())
    }

    fn scalar_add(&self, scalar: Scalar) -> Self {
        Self(&self.0 + scalar)
    }

    fn scalar_mul(&self, scalar: Scalar) -> Self {
        Self(&self.0 * scalar)
    }

    fn scalar_sub(&self, scalar: Scalar) -> Self {
        Self(&self.0 - scalar)
    }

    fn scalar_div(&self, scalar: Scalar) -> Self {
        Self(&self.0 / scalar)
    }

    fn index(&self, row: usize, col: usize) -> Scalar {
        self.0[[row, col]]
    }

    fn index_mut(&mut self, row: usize, col: usize) -> &mut Scalar {
        &mut self.0[[row, col]]
    }

    fn square(&self) -> Self {
        Self(&self.0 * &self.0)
    }

    fn sum(&self) -> Scalar {
        self.0.sum()
    }

    fn mean(&self) -> Scalar {
        self.sum() / (self.0.nrows() * self.0.ncols()) as Scalar
    }

    fn exp(&self) -> Self {
        Self(self.0.mapv(Scalar::exp))
    }

    fn maxof(&self, other: &Self) -> Self {
        let mat = Array2::from_shape_fn(
            (
                self.dim().0.min(other.dim().0),
                self.dim().1.min(other.dim().1),
            ),
            |(i, j)| self.0[[i, j]].max(other.0[[i, j]]),
        );
        Self(mat)
    }

    fn sign(&self) -> Self {
        Self(self.0.mapv(Scalar::signum))
    }

    fn minof(&self, other: &Self) -> Self {
        let mat = Array2::from_shape_fn(
            (
                self.dim().0.min(other.dim().0),
                self.dim().1.min(other.dim().1),
            ),
            |(i, j)| self.0[[i, j]].min(other.0[[i, j]]),
        );
        Self(mat)
    }

    fn log(&self) -> Self {
        Self(self.0.mapv(Scalar::ln))
    }

    fn sqrt(&self) -> Self {
        Self(self.0.mapv(Scalar::sqrt))
    }

    fn max(&self) -> Scalar {
        self.0.fold(Scalar::MIN, |max, x| max.max(*x))
    }

    fn min(&self) -> Scalar {
        self.0.fold(Scalar::MAX, |min, x| min.min(*x))
    }
}

impl Matrix {
    pub fn print(&self) {
        println!("{:?}", self.0);
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix {}x{}:\n", self.0.nrows(), self.0.ncols())?;
        for i in 0..self.0.nrows() {
            for j in 0..self.0.ncols() {
                write!(f, "{}\t", self.0[[i, j]])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
