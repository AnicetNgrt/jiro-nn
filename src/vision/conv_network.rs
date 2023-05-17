use std::fmt::Debug;

use crate::{
    layer::{DropoutLayer, Layer, LearnableLayer, ParameterableLayer},
    linalg::{Matrix, Scalar},
    network::NetworkLayer,
    vision::{image::Image, image::ImageTrait},
};

use super::image_layer::ImageLayer;

#[derive(Debug)]
pub struct ConvNetwork {
    layers: Vec<Box<dyn ConvNetworkLayer>>,
    channels: usize,
    out_channels: Option<usize>
}

impl ConvNetwork {
    pub fn new(layers: Vec<Box<dyn ConvNetworkLayer>>, channels: usize) -> Self {
        Self { layers, channels, out_channels: None }
    }
}

impl Layer for ConvNetwork {
    fn forward(&mut self, input: Matrix) -> Matrix {
        let mut output = Image::from_samples(&input, self.channels);

        for (_, layer) in self.layers.iter_mut().enumerate() {
            //println!("Layer {}", i);
            //println!("Input: {:?} : {}", output.image_dims(), output.samples());
            output = layer.forward(output);
            //println!("Output: {:?} : {}", output.image_dims(), output.samples());
        }

        self.out_channels = Some(output.channels());

        let out = output.flatten();
        out
    }

    fn backward(&mut self, epoch: usize, error_gradient: Matrix) -> Matrix {
        let mut error_gradient = Image::from_samples(&error_gradient, self.out_channels.unwrap());
        for layer in self.layers.iter_mut().rev() {
            error_gradient = layer.backward(epoch, error_gradient);
        }
        let grad = error_gradient.flatten();
        grad
    }
}

impl NetworkLayer for ConvNetwork {}

impl ParameterableLayer for ConvNetwork {
    fn as_learnable_layer(&self) -> Option<&dyn LearnableLayer> {
        Some(self)
    }

    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn LearnableLayer> {
        Some(self)
    }

    fn as_dropout_layer(&mut self) -> Option<&mut dyn DropoutLayer> {
        Some(self)
    }
}

impl LearnableLayer for ConvNetwork {
    fn get_learnable_parameters(&self) -> Vec<Vec<Scalar>> {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            layer.as_learnable_layer().map(|l| {
                params.append(&mut l.get_learnable_parameters());
            });
            params.push(vec![-1.0; 1]);
        }
        params
    }

    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Scalar>>) {
        // each layer's params are split in the params matrix at some line with a -1.0
        let mut params_matrix = params_matrix.clone();
        for layer in self.layers.iter_mut() {
            let mut layer_params = Vec::new();
            while let Some(param) = params_matrix.pop() {
                if param[0] == -1.0 {
                    break;
                }
                layer_params.push(param);
            }
            layer.as_learnable_layer_mut().map(|l| {
                l.set_learnable_parameters(&layer_params);
            });
        }
    }
}

impl DropoutLayer for ConvNetwork {
    fn enable_dropout(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| l.enable_dropout());
        });
    }

    fn disable_dropout(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.as_dropout_layer().map(|l| l.disable_dropout());
        });
    }
}

pub trait ConvNetworkLayer: ImageLayer + ParameterableLayer + Debug + Send {}
