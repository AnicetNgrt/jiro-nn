use arrayfire::print;
use neural_networks_rust::{vision::{Image, image::ImageTrait}, linalg::{Matrix, MatrixTrait}};

pub fn main() {
    let image = Image::from_samples(
        &Matrix::from_column_leading_matrix(&vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![3.0, 2.0, 0.0, 2.0],
            vec![6.0, 3.0, 1.0, 7.0]
        ]),
        1
    );

    print(&image.0);

    let matrix = vec![
        vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0]
        ],
        vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0]
        ],
        vec![
            vec![9.0, 8.0],
            vec![7.0, 6.0]
        ]
    ];

    let image = Image::from_fn(2, 2, 1, 3, |x, y, z, s| {
        println!("{} {} {} {}", x, y, z, s);
        matrix[s][y][x]
    });

    print(&image.0);

    let matrix = vec![
        vec![
            vec![
                vec![11.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![110.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![1100.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ]
        ],
        vec![
            vec![
                vec![21.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![210.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![2100.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ]
        ],
        vec![
            vec![
                vec![31.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![310.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![3100.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ]
        ],
        vec![
            vec![
                vec![41.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![410.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ],
            vec![
                vec![4100.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0]
            ]
        ],
    ];

    let image = Image::from_fn(2, 3, 3, 4, |x, y, z, s| {
        //println!("{} {} {} {}", x, y, z, s);
        matrix[s][z][y][x]
    });

    print(&image.0);
}