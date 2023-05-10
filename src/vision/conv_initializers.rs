use serde::{Deserialize, Serialize};

use super::image::Image;
use crate::{linalg::Scalar, vision::image::ImageTrait};

#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum ConvInitializers {
    Zeros,
    Uniform,
    UniformSigned,
    GlorotUniform,
}

impl ConvInitializers {
    pub fn gen_image(&self, nrow: usize, ncol: usize, nchan: usize, nsample: usize) -> Image {
        match self {
            ConvInitializers::Zeros => Image::zeros(nrow, ncol, nchan, nsample),
            ConvInitializers::Uniform => {
                Image::random_uniform(nrow, ncol, nchan, nsample, 0.0, 1.0)
            }
            ConvInitializers::UniformSigned => {
                Image::random_uniform(nrow, ncol, nchan, nsample, -1.0, 1.0)
            }
            ConvInitializers::GlorotUniform => {
                let limit = (6. / (ncol * nrow + ncol * nrow) as Scalar).sqrt();
                Image::random_uniform(nrow, ncol, nchan, nsample, -limit, limit)
            }
        }
    }
}
