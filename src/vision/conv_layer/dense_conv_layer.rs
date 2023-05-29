use crate::{
    layer::LearnableLayer,
    linalg::{Matrix, MatrixTrait, Scalar},
    vision::{
        conv_initializers::ConvInitializers, conv_optimizer::ConvOptimizers, image::Image,
        image::ImageTrait,
    },
};

use crate::vision::image_layer::ImageLayer;

use super::ConvLayer;

#[derive(Debug)]
pub struct DenseConvLayer {
    pub kernels: Image,
    biases: Image,
    input: Option<Image>,
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
            biases: biases_initializer.gen_image(1, 1, nkern, 1),
            input: None,
            kernels_optimizer,
            biases_optimizer,
        }
    }

    pub fn out_img_dims_and_channels(
        in_rows: usize,
        in_cols: usize,
        in_chans: usize,
        krows: usize,
        kcols: usize,
        kchans: usize,
    ) -> (usize, usize, usize) {
        let out_rows = in_rows - krows + 1;
        let out_cols = in_cols - kcols + 1;
        let out_chans = kchans;
        (out_rows, out_cols, out_chans)
    }
}

impl ImageLayer for DenseConvLayer {
    fn forward(&mut self, input: Image) -> Image {
        let res = input
            .cross_correlate(&self.kernels);

        if self.biases.image_dims() != res.image_dims() {
            self.biases = self.biases.tile(res.image_dims().0, res.image_dims().1, 1, 1);
        }

        let res = res
            .component_add(&self.biases);

        self.input = Some(input);
        res
    }

    fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image {
        let input = self.input.as_ref().unwrap();
        
        let mut input_grad_channels = vec![];
        for i in 0..input.channels() {
            let mut sum = Image::zeros(input.image_dims().0, input.image_dims().1, 1, input.samples());
            for k in 0..output_gradient.channels() {
                let kernel = self.kernels.get_sample(k).get_channel(i);
                let k_output_grad = output_gradient.get_channel_across_samples(k);
                let correlated = k_output_grad.convolve_full(&kernel);
                sum = sum.component_add(&correlated);
            }
            input_grad_channels.push(sum);
        }
        let input_grad = Image::join_channels(input_grad_channels);

        let mut kern_grad_samples = vec![];
        for k in 0..self.kernels.samples() {
            let mut kern_grad_channels = vec![];
            let output_grad_k = output_gradient.get_channel_across_samples(k).sum_samples();
            for i in 0..self.kernels.channels() {
                let input_i = input.get_channel_across_samples(i);
                let correlated = input_i.cross_correlate(&output_grad_k);
                kern_grad_channels.push(correlated.sum_samples());
            }
            let kern_grad_sample = Image::join_channels(kern_grad_channels);
            kern_grad_samples.push(kern_grad_sample);
        }
        let kern_grad = Image::join_samples(kern_grad_samples);

        let mut biases_grad_channels = vec![];
        for c in 0..self.biases.channels() {
            let channel = output_gradient.get_channel_across_samples(c);
            let channel = channel.sum_samples();
            biases_grad_channels.push(channel);
        }
        let biases_grad = Image::join_channels(biases_grad_channels);

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
        self.kernels = Image::from_samples(
            &Matrix::from_column_leading_matrix(&kernels),
            self.kernels.channels(),
        );
        self.biases = Image::from_samples(
            &Matrix::from_column_vector(&biases),
            self.biases.channels(),
        );
    }
}

impl ConvLayer for DenseConvLayer {
    fn scale_kernels(&mut self, scale: Scalar) {
        self.kernels = self.kernels.scalar_mul(scale);
    }
}
