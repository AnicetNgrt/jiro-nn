use crate::{linalg::{Scalar, MatrixTrait, Matrix}, vision::{
    conv_initializers::ConvInitializers, conv_optimizer::ConvOptimizers, image::Image,
    image::ImageTrait,
}, layer::LearnableLayer};

use super::ConvLayer;

#[derive(Debug)]
pub struct DenseConvLayer {
    pub kernels: Image,
    biases: Image,
    input: Option<Image>,
    output: Option<Image>,
    kernels_optimizer: ConvOptimizers,
    biases_optimizer: ConvOptimizers,
}

impl DenseConvLayer {
    pub fn new(
        nrow: usize,
        ncol: usize,
        nchan: usize,
        nkern: usize,
        kernels_initializer: ConvInitializers,
        biases_initializer: ConvInitializers,
        kernels_optimizer: ConvOptimizers,
        biases_optimizer: ConvOptimizers,
    ) -> Self {
        Self {
            kernels: kernels_initializer.gen_image(nrow, ncol, nchan, nkern),
            biases: biases_initializer.gen_image(nrow, ncol, nchan, 1),
            input: None,
            output: None,
            kernels_optimizer,
            biases_optimizer,
        }
    }
}

impl ConvLayer for DenseConvLayer {
    fn forward(&mut self, input: Image) -> Image {
        let res = input
            .cross_correlate(&self.kernels)
            .component_add(&self.biases);
        self.input = Some(input);
        self.output = Some(res.clone());
        res
    }

    fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image {
        let input_grad = output_gradient.convolve_full(&self.kernels);

        let kern_grad = self
            .input
            .as_ref()
            .unwrap()
            .cross_correlate(&output_gradient);

        let mut biases_grad = output_gradient.get_channel(0);
        for c in 1..output_gradient.image_dims().2 {
            biases_grad = biases_grad.component_add(&output_gradient.get_channel(c));
        }

        self.kernels = self
            .kernels_optimizer
            .update_parameters(epoch, &self.kernels, &kern_grad);
        self.biases = self
            .biases_optimizer
            .update_parameters(epoch, &self.biases, &biases_grad);
        input_grad
    }
}

impl LearnableLayer for DenseConvLayer {
    fn get_learnable_parameters(&self) -> Vec<Vec<Scalar>> {
        let mut params = self.kernels.flatten().get_data();
        params.push(self.biases.flatten().get_column(0));
        params
    }

    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Scalar>>) {
        let mut kernels = params_matrix.clone();
        let biases = kernels.pop().unwrap();
        self.kernels = Image::from_samples(&Matrix::from_column_leading_matrix(&kernels), self.kernels.image_dims().2);
        self.biases = Image::from_samples(&Matrix::from_column_vector(&biases), self.biases.image_dims().2);
    }
}
