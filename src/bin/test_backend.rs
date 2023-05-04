#[cfg(feature = "arrayfire")]
use arrayfire::get_available_backends;
use neural_networks_rust::linalg::{Matrix, MatrixTrait, Scalar};

pub fn print_mat(m: &Matrix) {
    println!("[");
    for row in m.get_data_row_leading() {
        print!("    ");
        for val in row {
            print!("{}, ", val);
        }
        println!("")
    }
    println!("]");
}

pub fn main() {
    #[cfg(feature = "arrayfire")]
    let backends = get_available_backends();
    #[cfg(feature = "arrayfire")]
    println!("{:#?}", backends);

    let m = Matrix::constant(10, 1000, 2.0);
    let m2 = Matrix::constant(1000, 10, 131.13313);
    let m3 = m.dot(&m2);

    print_mat(&m3);

    let m4 = m3.transpose();

    print_mat(&m4);

    let m5 = m4.columns_sum();

    print_mat(&m5);

    let m = Matrix::from_iter(1, 3, vec![1.0, 2.0, 3.0].into_iter());
    
    print_mat(&m);

    let m = Matrix::from_iter(3, 1, vec![1.0, 2.0, 3.0].into_iter());
    
    print_mat(&m);

    let m = Matrix::from_fn(3, 4, |i, _| { i as Scalar });

    print_mat(&m);

    let m = Matrix::from_fn(3, 4, |_, j| { j as Scalar });

    print_mat(&m);

    let m = Matrix::from_fn(3, 4, |i, j| { i as Scalar + j as Scalar });

    print_mat(&m);

    let m = Matrix::from_row_leading_matrix(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    
    print_mat(&m);

    let m = Matrix::from_column_leading_matrix(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    
    print_mat(&m);

    let m = Matrix::random_uniform(10, 5, 3.0, 18.4);

    print_mat(&m);

    let m = Matrix::random_normal(10, 5, 10., 2.0);

    print_mat(&m);
}