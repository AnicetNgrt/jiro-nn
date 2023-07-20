#[cfg(feature = "arrayfire")]
use arrayfire::{convolve3, flip, index, print, Seq};

#[allow(unused_imports)]
#[allow(unused_variables)]
use jiro_nn::{
    linalg::{Matrix, MatrixTrait},
    vision::{image::Image, image::ImageTrait},
};

#[cfg(feature = "arrayfire")]
pub fn main() {
    let image = Image::from_samples(
        &Matrix::from_column_leading_vector2(&vec![
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
            vec![
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
            vec![
                1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        ]),
        3,
    );

    println!("Original pictures");
    print(&image.0);

    let kernel = Image::from_samples(
        &Matrix::from_column_leading_vector2(&vec![vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ]]),
        3,
    );
    println!("Kernels");
    print(&kernel.0);

    let rot_kern = flip(&flip(&kernel.0, 0), 1);
    println!("Kernels rotated");
    print(&rot_kern);

    let res = convolve3(
        &image.0,
        &rot_kern,
        arrayfire::ConvMode::DEFAULT,
        arrayfire::ConvDomain::AUTO,
    );
    println!("Result");
    print(&res);

    println!("Result cropped");
    let out_size = image.image_dims().0 - kernel.image_dims().0 + 1;
    let res = index(
        &res,
        &[
            Seq::new(0, (out_size - 1).try_into().unwrap(), 1),
            Seq::new(0, (out_size - 1).try_into().unwrap(), 1),
            Seq::new(0, (kernel.samples() - 1).try_into().unwrap(), 1),
            Seq::new(0, (image.samples() - 1).try_into().unwrap(), 1),
        ],
    );

    print(&res);
}

#[cfg(not(feature = "arrayfire"))]
pub fn main() {
    println!("This example requires the arrayfire feature to be enabled");
}
