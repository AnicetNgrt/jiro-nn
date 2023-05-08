use arrayfire::{convolve2, index, join_many, Array, Dim4, Seq, fft2, convolve2_nn, convolve2_gradient_nn, ConvGradientType};

use crate::linalg::{Matrix, MatrixTrait, Scalar};

use super::image::ImageTrait;

pub struct Image(Array<Scalar>);

impl ImageTrait for Image {
    fn from_samples(samples: &Matrix, channels: usize) -> Self {
        let image_size = samples.dim().0 / channels;
        let mut samples_flattened = vec![];
        for i in 0..samples.dim().1 {
            let mut sample = samples.get_column(i);
            samples_flattened.append(&mut sample);
        }

        Self(Array::new(
            samples_flattened.as_slice(),
            Dim4::new(&[
                image_size.try_into().unwrap(),
                image_size.try_into().unwrap(),
                channels.try_into().unwrap(),
                samples.dim().1.try_into().unwrap(),
            ]),
        ))
    }

    fn component_add(&self, other: &Self) -> Self {
        let samples = self.samples();
        let other_samples = other.samples();

        if samples == other_samples {
            Self(&self.0 + &other.0)
        } else {
            let mut res_samples = vec![];
            let other_sample = other.get_sample(0);
            for i in 0..samples {
                let sample = self.get_sample(i);
                res_samples.push(Self(sample.0 + &other_sample.0));
            }
            Self::join_samples(res_samples)
        }
    }

    fn component_sub(&self, other: &Self) -> Self {
        let samples = self.samples();
        let other_samples = other.samples();

        if samples == other_samples {
            Self(&self.0 - &other.0)
        } else {
            let mut res_samples = vec![];
            let other_sample = other.get_sample(0);
            for i in 0..samples {
                let sample = self.get_sample(i);
                res_samples.push(Self(sample.0 - &other_sample.0));
            }
            Self::join_samples(res_samples)
        }
    }

    fn component_mul(&self, other: &Self) -> Self {
        let samples = self.samples();
        let other_samples = other.samples();

        if samples == other_samples {
            Self(&self.0 * &other.0)
        } else {
            let mut res_samples = vec![];
            let other_sample = other.get_sample(0);
            for i in 0..samples {
                let sample = self.get_sample(i);
                res_samples.push(Self(sample.0 * &other_sample.0));
            }
            Self::join_samples(res_samples)
        }
    }

    fn component_div(&self, other: &Self) -> Self {
        let samples = self.samples();
        let other_samples = other.samples();

        if samples == other_samples {
            Self(&self.0 / &other.0)
        } else {
            let mut res_samples = vec![];
            let other_sample = other.get_sample(0);
            for i in 0..samples {
                let sample = self.get_sample(i);
                res_samples.push(Self(sample.0 / &other_sample.0));
            }
            Self::join_samples(res_samples)
        }
    }

    fn scalar_add(&self, scalar: Scalar) -> Self {
        Self(&self.0 + scalar)
    }

    fn scalar_sub(&self, scalar: Scalar) -> Self {
        Self(&self.0 - scalar)
    }

    fn scalar_mul(&self, scalar: Scalar) -> Self {
        Self(&self.0 * scalar)
    }

    fn scalar_div(&self, scalar: Scalar) -> Self {
        Self(&self.0 / scalar)
    }

    fn convolve_cnn(&self, kernels: &Self) -> Self {
        Self(convolve2_nn(
            &self.0, 
            &kernels.0, 
            Dim4::new(&[1, 1, 1, 1]),
            Dim4::default(), 
            Dim4::default(),
        ))
    }

    fn convolve_grad_cnn(&self, input: &Self, filter: &Self, output: &Self) -> (Self, Self, Self) {
        (
            Self(convolve2_gradient_nn(
                &self.0, 
                &input.0,
                &filter.0,
                &output.0,
                Dim4::new(&[1, 1, 1, 1]),
                Dim4::default(), 
                Dim4::default(),
                ConvGradientType::DATA
            )),
            Self(convolve2_gradient_nn(
                &self.0, 
                &input.0,
                &filter.0,
                &output.0,
                Dim4::new(&[1, 1, 1, 1]),
                Dim4::default(), 
                Dim4::default(),
                ConvGradientType::FILTER
            )),
            Self(convolve2_gradient_nn(
                &self.0, 
                &input.0,
                &filter.0,
                &output.0,
                Dim4::new(&[1, 1, 1, 1]),
                Dim4::default(), 
                Dim4::default(),
                ConvGradientType::BIAS
            ))
        )
    }

    fn flatten(&self) -> Matrix {
        todo!()
    }

    fn image_dims(&self) -> (usize, usize, usize) {
        (
            self.0.dims()[0] as usize,
            self.0.dims()[1] as usize,
            self.0.dims()[2] as usize,
        )
    }

    fn samples(&self) -> usize {
        self.0.dims()[3] as usize
    }

    fn get_sample(&self, sample: usize) -> Self {
        Self(index(
            &self.0,
            &[
                Seq::default(),
                Seq::default(),
                Seq::default(),
                Seq::new(sample.try_into().unwrap(), sample.try_into().unwrap(), 1),
            ],
        ))
    }

    fn get_channel(&self, channel: usize) -> Self {
        Self(index(
            &self.0,
            &[
                Seq::default(),
                Seq::default(),
                Seq::new(channel.try_into().unwrap(), channel.try_into().unwrap(), 1),
                Seq::new(0, 0, 1),
            ],
        ))
    }

    fn join_channels(channels: Vec<Self>) -> Self {
        let inner_channels = channels.iter().map(|el| &el.0).collect::<Vec<_>>();
        Self(join_many(2, inner_channels))
    }

    fn join_samples(samples: Vec<Self>) -> Self {
        let inner_samples = samples.iter().map(|el| &el.0).collect::<Vec<_>>();
        Self(join_many(3, inner_samples))
    }
}
