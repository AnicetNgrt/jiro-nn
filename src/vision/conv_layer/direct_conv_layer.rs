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
pub struct DirectConvLayer {
    pub kernels: Image,
    biases: Image,
    input: Option<Image>,
    kernels_optimizer: ConvOptimizers,
    biases_optimizer: ConvOptimizers,
}

impl DirectConvLayer {
    pub fn new(
        krows: usize,
        kcols: usize,
        in_chans: usize,
        kernels_initializer: ConvInitializers,
        biases_initializer: ConvInitializers,
        kernels_optimizer: ConvOptimizers,
        biases_optimizer: ConvOptimizers,
    ) -> Self {
        Self {
            kernels: kernels_initializer.gen_image(krows, kcols, in_chans, 1),
            biases: biases_initializer.gen_image(1, 1, in_chans, 1),
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
    ) -> (usize, usize, usize) {
        let out_rows = in_rows - krows + 1;
        let out_cols = in_cols - kcols + 1;
        (out_rows, out_cols, in_chans)
    }
}

impl ImageLayer for DirectConvLayer {
    fn forward(&mut self, input: Image) -> Image {
        let mut channels = vec![];
        for c in 0..input.channels() {
            let channel = input.get_channel_across_samples(c);

            let kernel = self.kernels.get_channel(c);

            let correlated = channel.cross_correlate(&kernel);

            if self.biases.image_dims().0 != correlated.image_dims().0 {
                self.biases =
                    self.biases
                        .tile(correlated.image_dims().0, correlated.image_dims().1, 1, 1);
            }

            let bias = self.biases.get_channel(c);

            let result_channel = correlated.component_add(&bias);
            channels.push(result_channel);
        }
        let res = Image::join_channels(channels);
        self.input = Some(input);
        res
    }

    fn backward(&mut self, epoch: usize, output_gradient: Image) -> Image {
        let input = self.input.as_ref().unwrap();

        let mut input_grad_channels = vec![];
        for i in 0..input.channels() {
            let kernel = self.kernels.get_channel(i);
            let output_grad_i = output_gradient.get_channel_across_samples(i);
            let correlated = output_grad_i.convolve_full(&kernel);
            input_grad_channels.push(correlated);
        }
        let input_grad = Image::join_channels(input_grad_channels);

        let mut kern_grad_channels = vec![];
        for i in 0..input.channels() {
            let output_grad_i = output_gradient.get_channel_across_samples(i).sum_samples();
            let input_i = input.get_channel_across_samples(i);
            let correlated = input_i.cross_correlate(&output_grad_i).sum_samples();
            kern_grad_channels.push(correlated);
        }
        let kern_grad = Image::join_channels(kern_grad_channels);

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

impl LearnableLayer for DirectConvLayer {
    fn get_learnable_parameters(&self) -> Vec<Vec<Scalar>> {
        let mut params = self.kernels.flatten().get_data_col_leading();
        params.push(self.biases.flatten().get_column(0));
        params
    }

    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Scalar>>) {
        let mut kernels = params_matrix.clone();
        let biases = kernels.pop().unwrap();
        self.kernels = Image::from_samples(
            &Matrix::from_column_leading_vector2(&kernels),
            self.kernels.channels(),
        );
        self.biases =
            Image::from_samples(&Matrix::from_column_vector(&biases), self.biases.channels());
    }
}

impl ConvLayer for DirectConvLayer {
    fn scale_kernels(&mut self, scale: Scalar) {
        self.kernels = self.kernels.scalar_mul(scale);
    }
}
