use std::{sync::{Mutex, Arc}};

use rand::Rng;
use rand_distr::Distribution;
use rayon::prelude::*;

use super::MatrixTrait;

#[derive(Debug, Clone)]
pub struct Matrix {
    nrow: usize,
    ncol: usize,
    data: Vec<Vec<f64>>,
}

impl MatrixTrait for Matrix {
    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self { nrow, ncol, data: vec![vec![0.0; nrow]; ncol] }
    }

    fn random_uniform(nrow: usize, ncol: usize, min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| rng.gen_range(min..max)).collect())
            .collect();
        Self { nrow, ncol, data }
    }

    fn random_normal(nrow: usize, ncol: usize, mean: f64, std_dev: f64) -> Self {
        let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
        let data: Vec<Vec<f64>> = (0..ncol)
            .map(|_| (0..nrow).map(|_| normal.sample(&mut rand::thread_rng())).collect())
            .collect();
        Self { nrow, ncol, data }
    }

    fn from_iter(nrow: usize, ncol: usize, data: impl Iterator<Item = f64>) -> Self {
        let data: Vec<f64> = data.collect();
        assert_eq!(data.len(), nrow * ncol);
        let data: Vec<Vec<f64>> = data.chunks(nrow).map(|slice| slice.to_vec()).collect();
        Self { nrow, ncol, data }
    }

    fn from_row_leading_matrix(m: &Vec<Vec<f64>>) -> Self {
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

    fn from_column_leading_matrix(m: &Vec<Vec<f64>>) -> Self {
        let ncol = m.len();
        let nrow = m[0].len();
        Self { nrow, ncol, data: m.clone() }
    }

    fn from_column_vector(v: &Vec<f64>) -> Self {
        Self { nrow: v.len(), ncol: 1, data: vec![v.clone()] }
    }

    fn from_row_vector(v: &Vec<f64>) -> Self {
        let mut result = vec![vec![0.0; v.len()]; 1];
        for (i, col) in v.iter().enumerate() {
            result[0][i] = *col;
        }
        Self { nrow: v.len(), ncol: 1, data: result }
    }

    fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> f64,
    {
        let mut result = vec![vec![0.0; nrows]; ncols];
        for i in 0..ncols {
            for j in 0..nrows {
                result[i][j] = f(i, j);
            }
        }
        Self { nrow: nrows, ncol: ncols, data: result }
    }
    
    fn get_column(&self, index: usize) -> Vec<f64> {
        self.data[index].clone()
    }

    fn get_row(&self, index: usize) -> Vec<f64> {
        let mut result = vec![0.0; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            result[i] = col[index];
        }
        result
    }

    fn columns_map(&self, f: impl Fn(usize, &Vec<f64>) -> Vec<f64>) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        for (i, col) in self.data.iter().enumerate() {
            result[i] = f(i, col);
        }
        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }


    fn map(&self, f: impl Fn(f64) -> f64 + Sync) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];

        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = f(self.data[i][j]);
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn map_indexed_mut(&mut self, f: impl Fn(usize, usize, f64) -> f64 + Sync) -> &mut Self {
        self.data.par_iter_mut().enumerate().for_each(|(j, col)| {
            col.iter_mut().enumerate().for_each(|(i, val)| {
                *val = f(i, j, *val);
            });
        });
        self
    }

    fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.ncol, other.nrow);
        let result = Arc::new(Mutex::new(Matrix::zeros(self.nrow, other.ncol)));
        
        let edited_result = result.clone();
        (0..self.nrow).into_par_iter().for_each(move |i| {
            for j in 0..other.ncol {
                let mut sum = 0.0;
                for k in 0..other.nrow {
                    sum += self.index(i, k) * other.index(k, j);
                }
                *edited_result.lock().unwrap().index_mut(i, j) += sum;
            }
        });

        let result = result.lock().unwrap().clone(); 
        result
    }

    fn columns_sum(&self) -> Vec<f64> {
        let mut result = vec![0.0; self.nrow];

        result.par_iter_mut().zip(&self.data).for_each(|(row_res, col_data)| {
            col_data.iter().for_each(|val| {
                *row_res += val;
            });
        });

        result
    }

    fn component_mul(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];

        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] * other.data[i][j];
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn component_add(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];

        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] + other.data[i][j];
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn component_sub(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];

        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] - other.data[i][j];
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn component_div(&self, other: &Self) -> Self {
        assert_eq!(self.nrow, other.nrow);
        assert_eq!(self.ncol, other.ncol);
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];

        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] / other.data[i][j];
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn transpose(&self) -> Self {
        let mut result = vec![vec![0.0; self.ncol]; self.nrow];

        result.par_iter_mut().enumerate().for_each(|(j, col)| {
            col.iter_mut().enumerate().for_each(|(i, val)| {
                *val = self.data[i][j];
            });
        });

        Self { nrow: self.ncol, ncol: self.nrow, data: result }
    }

    
    fn dim(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn scalar_add(&self, scalar: f64) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];

        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] + scalar;
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn scalar_mul(&self, scalar: f64) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        
        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] * scalar;
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn scalar_sub(&self, scalar: f64) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
        
        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] - scalar;
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn scalar_div(&self, scalar: f64) -> Self {
        let mut result = vec![vec![0.0; self.nrow]; self.ncol];
    
        result.par_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.data[i][j] / scalar;
            });
        });

        Self { nrow: self.nrow, ncol: self.ncol, data: result }
    }

    fn index(&self, row: usize, col: usize) -> f64 {
        self.data[col][row]
    }

    fn index_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        &mut self.data[col][row]
    }

    fn get_data(&self) -> Vec<Vec<f64>> {
        self.data.clone()
    }

    fn get_data_row_leading(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.ncol]; self.nrow];

        result.par_iter_mut().enumerate().for_each(|(j, col)| {
            col.iter_mut().enumerate().for_each(|(i, val)| {
                *val = self.data[i][j];
            });
        });

        result
    }
}
