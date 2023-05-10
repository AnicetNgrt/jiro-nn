use crate::{
    layer::{LearnableLayer, ParameterableLayer},
    linalg::{Scalar},
    vision::{
        image::Image,
        image::ImageTrait, conv_network::ConvNetworkLayer,
    },
};

use crate::vision::image_layer::ImageLayer;

#[derive(Debug)]
pub struct AvgPoolingLayer {
    pub div: usize,
}

impl AvgPoolingLayer {
    pub fn new(
        div: usize,
    ) -> Self {
        Self {
            div,
        }
    }
}

impl ImageLayer for AvgPoolingLayer {
    fn forward(&mut self, input: Image) -> Image {
        let unwrapped = input.unwrap(self.div, self.div, self.div, self.div, 0, 0);
        let meaned = unwrapped.mean_along(0);
        let result = meaned.wrap(input.image_dims().0/self.div, input.image_dims().1/self.div, 1, 1, 1, 1, 0, 0);
        result
    }

    fn backward(&mut self, _epoch: usize, output_gradient: Image) -> Image {
        let input_grad = output_gradient
            .scalar_div((self.div * self.div) as Scalar)
            .unwrap(1, 1, 1, 1, 0, 0)
            .tile(self.div * self.div, 1, 1, 1)
            .wrap(
                output_gradient.image_dims().0 * self.div,
                output_gradient.image_dims().1 * self.div,
                self.div,
                self.div,
                self.div,
                self.div,
                0,
                0,
            );
        
        input_grad
    }
}

impl ParameterableLayer for AvgPoolingLayer {
    fn as_learnable_layer(&self) -> Option<&dyn LearnableLayer> {
        None
    }

    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn LearnableLayer> {
        None
    }

    fn as_dropout_layer(&mut self) -> Option<&mut dyn crate::layer::DropoutLayer> {
        None
    }
}

impl ConvNetworkLayer for AvgPoolingLayer {
}