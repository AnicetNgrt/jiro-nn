pub mod nalgebra_matrix;

#[cfg(feature = "nalgebra_backend")]
pub type Matrix = nalgebra_matrix::Matrix;

pub mod matrix;

#[cfg(all(not(feature = "nalgebra_backend")))]
pub type Matrix = matrix::Matrix;
