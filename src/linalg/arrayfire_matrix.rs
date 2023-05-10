use core::fmt;

use arrayfire::{
    add, constant, div, exp, index, matmul, maxof, mean_all, minof, mul, pow, print,
    random_normal, random_uniform, sign, sqrt, sub, sum_all, transpose, Array, Dim4, MatProp,
    RandomEngine, Seq, moddims,
};
use rand::Rng;

use super::{MatrixTrait, Scalar};

/// Arrayfire matrix

#[derive(Clone)]
pub struct Matrix(pub Array<Scalar>);

impl MatrixTrait for Matrix {
    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self(constant!(0 as Scalar; nrow.try_into().unwrap(), ncol.try_into().unwrap()))
    }

    fn constant(nrow: usize, ncol: usize, value: Scalar) -> Self {
        Self(constant!(value; nrow.try_into().unwrap(), ncol.try_into().unwrap()))
    }

    /// Creates a matrix with random values between min and max (excluded).
    fn random_uniform(nrow: usize, ncol: usize, min: Scalar, max: Scalar) -> Self {
        let mut rng = rand::thread_rng();
        Self(
            random_uniform::<Scalar>(
                Dim4::new(&[nrow.try_into().unwrap(), ncol.try_into().unwrap(), 1, 1]),
                &RandomEngine::new(
                    arrayfire::RandomEngineType::MERSENNE_GP11213,
                    Some(rng.gen()),
                ),
            ) * (max - min)
                + constant!(min; nrow.try_into().unwrap(), ncol.try_into().unwrap()),
        )
    }

    /// Creates a matrix with random values following a normal distribution.
    fn random_normal(nrow: usize, ncol: usize, mean: Scalar, std_dev: Scalar) -> Self {
        let mut rng = rand::thread_rng();
        Self(
            random_normal::<Scalar>(
                Dim4::new(&[nrow.try_into().unwrap(), ncol.try_into().unwrap(), 1, 1]),
                &RandomEngine::new(
                    arrayfire::RandomEngineType::MERSENNE_GP11213,
                    Some(rng.gen()),
                ),
            ) * std_dev
                + constant!(mean; nrow.try_into().unwrap(), ncol.try_into().unwrap()),
        )
    }

    /// Fills the matrix with the iterator columns after columns by chunking the data by n_rows.
    /// ```txt
    /// Your data : [[col1: row0 row1 ... rowNrow][col2]...[colNcol]]
    /// ```
    fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = Scalar>) -> Self {
        let data: Vec<Scalar> = data.collect();
        assert_eq!(data.len(), nrow * ncol);

        Self(Array::new_strided(
            data.as_slice(),
            0,
            Dim4::new(&[nrow.try_into().unwrap(), ncol.try_into().unwrap(), 1, 1]),
            Dim4::new(&[1, nrow.try_into().unwrap(), u64::MAX, u64::MAX]),
        ))
    }

    fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> Scalar,
    {
        Self(Array::new_strided(
            (0..nrows * ncols)
                .map(|i| f(i % nrows, i / nrows))
                .collect::<Vec<_>>()
                .as_slice(),
            0,
            Dim4::new(&[nrows.try_into().unwrap(), ncols.try_into().unwrap(), 1, 1]),
            Dim4::new(&[1, nrows.try_into().unwrap(), u64::MAX, u64::MAX]),
        ))
    }

    /// ```txt
    /// Your data :
    /// [
    ///    [row0: col0 col1 ... colNcol],
    ///    [row1: col0 col1 ... colNcol],
    ///     ...
    ///    [rowNrow: col0 col1 ... colNcol],
    /// ]
    /// ```
    fn from_row_leading_matrix(m: &Vec<Vec<Scalar>>) -> Self {
        let mat = Self::from_column_leading_matrix(m);
        mat.transpose()
    }

    fn from_column_leading_matrix(m: &Vec<Vec<Scalar>>) -> Self {
        let ncol = m.len();
        let nrow = m[0].len();
        Self(Array::new_strided(
            m.concat().as_slice(),
            0,
            Dim4::new(&[nrow.try_into().unwrap(), ncol.try_into().unwrap(), 1, 1]),
            Dim4::new(&[1, nrow.try_into().unwrap(), u64::MAX, u64::MAX]),
        ))
    }

    /// fills a column vector row by row with values of index 0 to v.len()
    fn from_column_vector(v: &Vec<Scalar>) -> Self {
        Self(Array::new_strided(
            v.as_slice(),
            0,
            Dim4::new(&[v.len().try_into().unwrap(), 1, 1, 1]),
            Dim4::new(&[1, v.len().try_into().unwrap(), u64::MAX, u64::MAX]),
        ))
    }

    /// fills a row vector column by column with values of index 0 to v.len()
    fn from_row_vector(v: &Vec<Scalar>) -> Self {
        Self(Array::new_strided(
            v.as_slice(),
            0,
            Dim4::new(&[1, v.len().try_into().unwrap(), 1, 1]),
            Dim4::new(&[v.len().try_into().unwrap(), 1, u64::MAX, u64::MAX]),
        ))
    }

    fn get_column(&self, idx: usize) -> Vec<Scalar> {
        let res = index(
            &self.0,
            &[Seq::default(), Seq::new(idx as u32, idx as u32, 1)],
        );
        let mut buffer = Vec::<Scalar>::new();
        buffer.resize(self.dim().0, 0.0);
        res.host(&mut buffer);
        buffer
    }

    fn get_row(&self, idx: usize) -> Vec<Scalar> {
        let res = index(
            &self.0,
            &[Seq::new(idx as u32, idx as u32, 1), Seq::default()],
        );
        let mut buffer = Vec::<Scalar>::new();
        buffer.resize(self.dim().1, 0.0);
        res.host(&mut buffer);
        buffer
    }

    fn columns_map(&self, _f: impl Fn(usize, &Vec<Scalar>) -> Vec<Scalar>) -> Self {
        unimplemented!("Deprecated")
    }

    fn map_indexed_mut(&mut self, _f: impl Fn(usize, usize, Scalar) -> Scalar + Sync) -> &mut Self {
        unimplemented!("Deprecated")
    }

    fn map(&self, _f: impl Fn(Scalar) -> Scalar + Sync) -> Self {
        unimplemented!("Deprecated")
    }

    fn dot(&self, other: &Self) -> Self {
        let res = matmul(&self.0, &other.0, MatProp::NONE, MatProp::NONE);
        Self(res)
    }

    fn columns_sum(&self) -> Self {
        let res = matmul(
            &self.0,
            &constant!(1.0; self.0.dims()[1]),
            MatProp::NONE,
            MatProp::NONE,
        );
        Self(res)
    }

    fn component_mul(&self, other: &Self) -> Self {
        Self(mul(&self.0, &other.0, false))
    }

    fn component_add(&self, other: &Self) -> Self {
        Self(add(&self.0, &other.0, false))
    }

    fn component_sub(&self, other: &Self) -> Self {
        Self(sub(&self.0, &other.0, false))
    }

    fn component_div(&self, other: &Self) -> Self {
        Self(div(&self.0, &other.0, false))
    }

    fn transpose(&self) -> Self {
        Self(transpose(&self.0, false))
    }

    fn get_data(&self) -> Vec<Vec<Scalar>> {
        let mut cols = Vec::new();
        for i in 0..self.dim().1 {
            cols.push(self.get_column(i));
        }
        cols
    }

    fn get_data_row_leading(&self) -> Vec<Vec<Scalar>> {
        let mut rows = Vec::new();
        for i in 0..self.dim().0 {
            rows.push(self.get_row(i));
        }
        rows
    }

    /// returns the dimensions of the matrix (nrow, ncol)
    fn dim(&self) -> (usize, usize) {
        (self.0.dims()[0] as usize, self.0.dims()[1] as usize)
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

    fn index(&self, _row: usize, _col: usize) -> Scalar {
        unimplemented!("Deprecated")
    }

    fn index_mut(&mut self, _row: usize, _col: usize) -> &mut Scalar {
        unimplemented!("Deprecated")
    }

    fn square(&self) -> Self {
        Self(pow(
            &self.0,
            &constant!(2.0 as Scalar; self.dim().0.try_into().unwrap(), self.dim().1.try_into().unwrap(), 1, 1),
            false,
        ))
    }

    fn sum(&self) -> Scalar {
        sum_all(&self.0).0
    }

    fn mean(&self) -> Scalar {
        mean_all(&self.0).0 as Scalar
    }

    fn exp(&self) -> Self {
        Self(exp(&self.0))
    }

    fn maxof(&self, other: &Self) -> Self {
        Self(maxof(&self.0, &other.0, false))
    }

    fn sign(&self) -> Self {
        Self(sign(&self.0)).scalar_mul(-2.0).scalar_add(1.0)
    }

    fn minof(&self, other: &Self) -> Self {
        Self(minof(&self.0, &other.0, false))
    }

    fn sqrt(&self) -> Self {
        Self(sqrt(&self.0))
    }
}

impl Matrix {
    pub fn print(&self) {
        print(&self.0)
    }

    pub fn from_array(size: usize, samples: usize, array: &Array<Scalar>) -> Self {
        Matrix(moddims(array,
            Dim4::new(&[
                size.try_into().unwrap(),
                samples.try_into().unwrap(),
                1,
                1
            ])
        ))
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::new();
        for row in self.get_data() {
            result.push_str(&format!("{:?}\n", row));
        }

        write!(f, "{}", result)
    }
}
