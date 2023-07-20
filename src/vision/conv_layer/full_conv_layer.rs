use std::cmp::Ordering;

use rand::Rng;

use crate::layer::{DropoutLayer, LearnableLayer, ParameterableLayer};
use crate::linalg::Scalar;
use crate::vision::conv_network::ConvNetworkLayer;

use super::{ConvLayer, Image};
use crate::vision::conv_activation::ConvActivationLayer;
use crate::vision::image::ImageTrait;
use crate::vision::image_layer::ImageLayer;

#[derive(Debug)]
pub struct FullConvLayer {
    conv: Box<dyn ConvLayer>,
    activation: ConvActivationLayer,
    dropout_enabled: bool,
    dropout_rate: Option<Scalar>,
    mask: Option<Image>,
}

impl FullConvLayer {
    pub fn new(
        conv: Box<dyn ConvLayer>,
        activation: ConvActivationLayer,
        dropout: Option<Scalar>,
    ) -> Self {
        Self {
            conv,
            activation,
            dropout_rate: dropout,
            dropout_enabled: false,
            mask: None,
        }
    }

    fn generate_dropout_mask(
        &mut self,
        kern_size: (usize, usize, usize),
        nkern: usize,
    ) -> Option<(Image, Scalar)> {
        if let Some(dropout_rate) = self.dropout_rate {
            let mut rng = rand::thread_rng();
            let dropout_mask = Image::from_fn(
                kern_size.0,
                kern_size.1,
                kern_size.2,
                nkern,
                |_, _, _, _| {
                    if rng
                        .gen_range((0.0 as Scalar)..(1.0 as Scalar))
                        .total_cmp(&self.dropout_rate.unwrap())
                        == Ordering::Greater
                    {
                        1.0
                    } else {
                        0.0
                    }
                },
            );
            Some((dropout_mask, dropout_rate))
        } else {
            None
        }
    }
}

impl ImageLayer for FullConvLayer {
    fn forward(&mut self, mut input: Image) -> Image {
        let output = if self.dropout_enabled {
            if let Some((mask, _)) = self.generate_dropout_mask(input.image_dims(), input.samples())
            {
                input = input.component_mul(&mask);
                self.mask = Some(mask);
            };
            self.conv.forward(input)
        } else {
            if let Some(dropout_rate) = self.dropout_rate {
                self.conv.scale_kernels(1.0 - dropout_rate);
                let output = self.conv.forward(input);
                self.conv.scale_kernels(1.0 / (1.0 - dropout_rate));
                output
            } else {
                self.conv.forward(input)
            }
        };

        self.activation.forward(output)
    }

    fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image {
        let activation_input_gradient = self.activation.backward(epoch, output_gradient);
        let input_gradient = self.conv.backward(epoch, activation_input_gradient);

        if let Some(mask) = &self.mask {
            input_gradient.component_mul(&mask)
        } else {
            input_gradient
        }
    }
}

impl LearnableLayer for FullConvLayer {
    fn get_learnable_parameters(&self) -> Vec<Vec<Scalar>> {
        self.conv.get_learnable_parameters()
    }

    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Scalar>>) {
        self.conv.set_learnable_parameters(params_matrix)
    }
}

impl DropoutLayer for FullConvLayer {
    fn enable_dropout(&mut self) {
        self.dropout_enabled = true;
    }

    fn disable_dropout(&mut self) {
        self.dropout_enabled = false;
    }
}

impl ParameterableLayer for FullConvLayer {
    fn as_learnable_layer(&self) -> Option<&dyn crate::layer::LearnableLayer> {
        Some(self)
    }

    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn crate::layer::LearnableLayer> {
        Some(self)
    }

    fn as_dropout_layer(&mut self) -> Option<&mut dyn crate::layer::DropoutLayer> {
        Some(self)
    }
}

impl ConvNetworkLayer for FullConvLayer {}
