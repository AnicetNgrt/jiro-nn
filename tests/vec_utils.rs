#[macro_use]
extern crate assert_float_eq;

use jiro_nn::vec_utils::*;

#[test]
fn test_avg_vector_empty() {
    let vec = vec![];
    assert!(avg_vector(&vec).is_nan());
}

#[test]
fn test_avg_vector_single() {
    let vec = vec![42.0];
    assert_eq!(avg_vector(&vec), 42.0);
}

#[test]
fn test_avg_vector_two() {
    let vec = vec![2.0, 4.0];
    assert_eq!(avg_vector(&vec), 3.0);
}

#[test]
fn test_avg_vector_negative() {
    let vec = vec![-1.0, 0.0, 1.0];
    assert_eq!(avg_vector(&vec), 0.0);
}

#[test]
fn test_avg_vector_fractional() {
    let vec = vec![1.5, 2.5, 3.5];
    assert_eq!(avg_vector(&vec), 2.5);
}

#[test]
fn test_median_vector_empty() {
    let vec = vec![];
    assert!(median_vector(&vec).is_nan());
}

#[test]
fn test_median_vector_single() {
    let vec = vec![42.0];
    assert_eq!(median_vector(&vec), 42.0);
}

#[test]
fn test_median_vector_two() {
    let vec = vec![2.0, 4.0];
    assert_eq!(median_vector(&vec), 3.0);
}

#[test]
fn test_median_vector_negative() {
    let vec = vec![-1.0, 0.0, 1.0];
    assert_eq!(median_vector(&vec), 0.0);
}

#[test]
fn test_median_vector_fractional() {
    let vec = vec![1.5, 2.5, 3.5];
    assert_eq!(median_vector(&vec), 2.5);
}

#[test]
fn test_median_vector_even() {
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(median_vector(&vec), 2.5);
}

#[test]
fn test_quartiles_vector_even_len() {
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    let (q1, q2, q3) = quartiles_vector(&vec);
    assert_float_relative_eq!(q1, 1.5, 0.00001);
    assert_float_relative_eq!(q2, 2.5, 0.00001);
    assert_float_relative_eq!(q3, 3.5, 0.00001);
}

#[test]
fn test_quartiles_vector_odd_len() {
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (q1, q2, q3) = quartiles_vector(&vec);
    assert_float_relative_eq!(q1, 1.5, 0.00001);
    assert_float_relative_eq!(q2, 3.0, 0.00001);
    assert_float_relative_eq!(q3, 3.5, 0.00001);
}

#[test]
fn test_quartiles_vector_duplicate_values() {
    let vec = vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0];
    let (q1, q2, q3) = quartiles_vector(&vec);
    println!("{} {} {}", q1, q2, q3);
    assert_float_relative_eq!(q1, 2.0, 0.00001);
    assert_float_relative_eq!(q2, 3.5, 0.00001);
    assert_float_relative_eq!(q3, 4.0, 0.00001);
}

#[test]
fn test_vector_quartiles_iqr_odd() {
    let vals = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let (q1, q2, q3, min, max) = vector_quartiles_iqr(&vals);
    assert_float_relative_eq!(q1, 2.0, 0.00001);
    assert_float_relative_eq!(q2, 5.0, 0.00001);
    assert_float_relative_eq!(q3, 6.0, 0.00001);
    assert_float_relative_eq!(min, -4.0, 0.00001);
    assert_float_relative_eq!(max, 12.0, 0.00001);
}

#[test]
fn test_vector_quartiles_iqr_even() {
    let vals = vec![1.0, 3.0, 5.0, 7.0];
    let (q1, q2, q3, min, max) = vector_quartiles_iqr(&vals);
    println!("{} {} {}", q1, q2, q3);
    assert_float_relative_eq!(q1, 2.0, 0.00001);
    assert_float_relative_eq!(q2, 4.0, 0.00001);
    assert_float_relative_eq!(q3, 6.0, 0.00001);
    assert_float_relative_eq!(min, -4.0, 0.00001);
    assert_float_relative_eq!(max, 12.0, 0.00001);
}

#[test]
fn test_normalize_vector() {
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    let (normalized_vec, min, max) = normalize_vector(&vec);
    assert_float_relative_eq!(normalized_vec[0], 0.0, 0.00001);
    assert_float_relative_eq!(normalized_vec[1], 0.33333, 0.00001);
    assert_float_relative_eq!(normalized_vec[2], 0.66666, 0.00001);
    assert_float_relative_eq!(normalized_vec[3], 1.0, 0.00001);
    assert_float_relative_eq!(min, 1.0, 0.00001);
    assert_float_relative_eq!(max, 4.0, 0.00001);
}

#[test]
fn test_denormalize_vector() {
    let vec = vec![0.0, 0.33333, 0.66666, 1.0];
    let denormalized_vec = denormalize_vector(&vec, 1.0, 4.0);
    assert_float_relative_eq!(denormalized_vec[0], 1.0, 0.00001);
    assert_float_relative_eq!(denormalized_vec[1], 2.0, 0.00001);
    assert_float_relative_eq!(denormalized_vec[2], 3.0, 0.00001);
    assert_float_relative_eq!(denormalized_vec[3], 4.0, 0.00001);
}

#[test]
fn test_normalize_vec2() {
    let vec = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let (normalized_vec, min, max) = normalize_vec2(&vec);
    assert_float_relative_eq!(normalized_vec[0][0], 0.0, 0.00001);
    assert_float_relative_eq!(normalized_vec[0][1], 0.125, 0.00001);
    assert_float_relative_eq!(normalized_vec[0][2], 0.25, 0.00001);
    assert_float_relative_eq!(normalized_vec[1][0], 0.375, 0.00001);
    assert_float_relative_eq!(normalized_vec[1][1], 0.5, 0.00001);
    assert_float_relative_eq!(normalized_vec[1][2], 0.625, 0.00001);
    assert_float_relative_eq!(normalized_vec[2][0], 0.75, 0.00001);
    assert_float_relative_eq!(normalized_vec[2][1], 0.875, 0.00001);
    assert_float_relative_eq!(normalized_vec[2][2], 1.0, 0.00001);
    assert_float_relative_eq!(min, 1.0, 0.00001);
    assert_float_relative_eq!(max, 9.0, 0.00001);
}

#[test]
fn test_denormalize_vec2() {
    let input = vec![vec![0.0, 0.25], vec![0.5, 0.75], vec![1.0, 1.0]];
    let min = 0.0;
    let max = 1.0;
    let expected_output = vec![
        vec![min, (0.25 * (max - min)) + min],
        vec![(0.5 * (max - min)) + min, (0.75 * (max - min)) + min],
        vec![max, max],
    ];
    let output = denormalize_vec2(&input, min, max);
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            assert_float_relative_eq!(output[i][j], expected_output[i][j], 0.00001);
        }
    }
}

#[test]
fn test_vectors_correlation() {
    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let corr = vectors_correlation(&vec1, &vec2);
    assert_float_relative_eq!(corr.unwrap(), 1.0, 0.00001);

    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let corr = vectors_correlation(&vec1, &vec2);
    assert_float_relative_eq!(corr.unwrap(), -1.0, 0.00001);

    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let corr = vectors_correlation(&vec1, &vec2);
    assert_float_relative_eq!(corr.unwrap(), 1.0, 0.00001);

    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![5.0, 4.0, 3.0, 4.0, 5.0];
    let corr = vectors_correlation(&vec1, &vec2);
    assert_float_relative_eq!(corr.unwrap(), 0.0, 0.00001);

    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![3.0, 3.0, 3.0, 4.0, 5.0];
    let corr = vectors_correlation(&vec1, &vec2);
    assert_float_relative_eq!(corr.unwrap(), 0.88388, 0.00001);
}
