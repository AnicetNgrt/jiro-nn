#[macro_use]
extern crate assert_float_eq;

use jiro_nn::linalg::{Matrix, MatrixTrait, Scalar};

#[test]
fn test_zeros() {
    let m = Matrix::zeros(2, 3);
    for row in m.get_data_row_leading() {
        for val in row {
            assert_float_relative_eq!(val, 0.0, 0.00001);
        }
    }
}

#[test]
fn test_random_uniform() {
    let mat = Matrix::random_uniform(2, 2, 0.0, 1.0);
    assert_eq!(mat.dim().0, 2);
    assert_eq!(mat.dim().1, 2);
    assert!(mat
        .get_data_row_leading()
        .iter()
        .all(|row| { row.iter().all(|&val| val >= 0.0 && val < 1.0) }));
}

#[test]
fn test_random_normal() {
    let mat = Matrix::random_normal(100, 100, 0.0, (1.0 as Scalar).sqrt());
    assert_eq!(mat.dim().0, 100);
    assert_eq!(mat.dim().1, 100);
    // Check that the values are roughly normally distributed
    let mean = mat.get_data_row_leading().iter().flatten().sum::<Scalar>() / 10000.0;
    let variance = mat
        .get_data_row_leading()
        .iter()
        .flatten()
        .map(|val| (val - mean).powi(2))
        .sum::<Scalar>()
        / 1000.0;

    assert!(mean.abs() < 0.1);
    assert!((variance.abs() - 1.0).abs() < 0.1);
}

#[test]
fn test_from_iter() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Matrix::from_iter(2, 3, data.iter().cloned());
    assert_float_relative_eq!(m.get_data_col_leading()[0][0], 1.0, 0.00001);
    assert_float_relative_eq!(m.get_data_col_leading()[2][1], 6.0, 0.00001);
}

#[test]
fn test_from_row_leading_vector2() {
    let m = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_float_relative_eq!(m.get_data_col_leading()[0][0], 1.0, 0.00001);
    assert_float_relative_eq!(m.get_data_col_leading()[2][1], 6.0, 0.00001);
}

#[test]
fn test_from_column_vector() {
    let m = Matrix::from_column_vector(&vec![1.0, 2.0, 3.0]);
    assert_float_relative_eq!(m.get_data_row_leading()[0][0], 1.0, 0.00001);
    assert_float_relative_eq!(m.get_data_row_leading()[2][0], 3.0, 0.00001);
}

#[test]
fn test_from_row_vector() {
    let m = Matrix::from_row_vector(&vec![1.0, 2.0, 3.0]);
    println!("{:?}", m.get_data_row_leading());
    assert_float_relative_eq!(m.get_data_row_leading()[0][0], 1.0, 0.00001);
    assert_float_relative_eq!(m.get_data_row_leading()[0][2], 3.0, 0.00001);
}

#[test]
fn test_get_column() {
    let m = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let col = m.get_column(1);
    assert_float_relative_eq!(col[0], 2.0, 0.00001);
    assert_float_relative_eq!(col[1], 5.0, 0.00001);
}

#[test]
fn test_get_row() {
    let m = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let row = m.get_row(1);
    assert_float_relative_eq!(row[0], 4.0, 0.00001);
    assert_float_relative_eq!(row[1], 5.0, 0.00001);
    assert_float_relative_eq!(row[2], 6.0, 0.00001);
}

// #[test]
// fn test_map() {
//     let m = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

//     let mapped = m.map(|x| x * 2.0);

//     assert_float_relative_eq!(mapped.get_data_row_leading()[0][0], 2.0, 0.00001);
//     assert_float_relative_eq!(mapped.get_data_row_leading()[1][2], 12.0, 0.00001);
// }

#[test]
fn test_dot() {
    let m1 = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 =
        Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    let result = m1.dot(&m2);
    assert_eq!(result.dim(), (2, 2));
    assert_float_relative_eq!(result.get_data_row_leading()[0][0], 22.0, 0.00001);
    assert_float_relative_eq!(result.get_data_row_leading()[1][1], 64.0, 0.00001);
}

#[test]
fn test_columns_sum() {
    let m = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let sums = m.columns_sum().get_data_row_leading();
    assert_float_relative_eq!(sums[0][0], 6.0, 0.00001);
    assert_float_relative_eq!(sums[1][0], 15.0, 0.00001);
}

#[test]
fn test_component_mul() {
    let m1 = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix::from_row_leading_vector2(&vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]);
    let result = m1.component_mul(&m2);
    assert_float_relative_eq!(result.get_data_row_leading()[0][0], 2.0, 0.00001);
    assert_float_relative_eq!(result.get_data_row_leading()[1][2], 42.0, 0.00001);
}

#[test]
fn test_component_add() {
    let m1 = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix::from_row_leading_vector2(&vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]);
    let result = m1.component_add(&m2);

    println!("{:#?}", result.get_data_row_leading());

    assert_float_relative_eq!(result.get_data_row_leading()[0][0], 3.0, 0.00001);
    assert_float_relative_eq!(result.get_data_row_leading()[1][2], 13.0, 0.00001);
}

#[test]
fn test_component_sub() {
    let m1 = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let m2 = Matrix::from_row_leading_vector2(&vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]);
    let result = m1.component_sub(&m2);
    assert_float_relative_eq!(result.get_data_row_leading()[0][0], -1.0, 0.00001);
    assert_float_relative_eq!(result.get_data_row_leading()[1][2], -1.0, 0.00001);
}

#[test]
fn test_component_div() {
    let m1 = Matrix::from_row_leading_vector2(&vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
    let m2 = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let result = m1.component_div(&m2);
    assert_float_relative_eq!(result.get_data_row_leading()[0][0], 2.0, 0.00001);
    assert_float_relative_eq!(result.get_data_row_leading()[1][2], 2.0, 0.00001);
}

#[test]
fn test_transpose() {
    let m = Matrix::from_row_leading_vector2(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let transposed = m.transpose();
    assert_float_relative_eq!(transposed.get_data_row_leading()[0][0], 1.0, 0.00001);
    assert_float_relative_eq!(transposed.get_data_row_leading()[2][1], 6.0, 0.00001);
}
